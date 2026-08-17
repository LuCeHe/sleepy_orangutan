// Diagonal linear scan h_t = λ_t * h_{t-1} + c_t (elementwise over hidden dim).
// Semantics aligned with diagonal_scan_triton.py: fp32 math, log-prefix form, LP_FLOOR = -85.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>
#include <cuda_runtime.h>

namespace {

__device__ __forceinline__ int64_t idx3(
    int64_t b,
    int64_t t,
    int64_t h,
    int64_t sb,
    int64_t st,
    int64_t sh) {
  return b * sb + t * st + h * sh;
}

// Inclusive scan (sum) for n=512 in shared memory using a Blelloch-style sweep.
// After calling, the array contains the INCLUSIVE prefix sum.
__device__ __forceinline__ void inclusive_scan_512(float* s) {
  // Copy original for inclusive reconstruction.
  __shared__ float orig[512];
  orig[threadIdx.x] = s[threadIdx.x];
  __syncthreads();

  // upsweep
  for (int offset = 1; offset < 512; offset <<= 1) {
    int i = (threadIdx.x + 1) * offset * 2 - 1;
    if (i < 512) {
      s[i] += s[i - offset];
    }
    __syncthreads();
  }
  // downsweep (exclusive)
  if (threadIdx.x == 0) {
    s[511] = 0.f;
  }
  __syncthreads();
  for (int offset = 256; offset >= 1; offset >>= 1) {
    int i = (threadIdx.x + 1) * offset * 2 - 1;
    if (i < 512) {
      float t = s[i - offset];
      s[i - offset] = s[i];
      s[i] += t;
    }
    __syncthreads();
  }
  // exclusive -> inclusive
  s[threadIdx.x] += orig[threadIdx.x];
  __syncthreads();
}

// v2: one block per (b,h) with 512 threads scanning time (T<=512).
__global__ void diagonal_scan_fwd_kernel_v2_512(
    const float* __restrict__ lam,
    const float* __restrict__ c,
    const float* __restrict__ h0,
    float* __restrict__ out,
    int B,
    int T,
    int H,
    int64_t sb,
    int64_t st,
    int64_t sh,
    int64_t h0_sb,
    int64_t h0_sh) {
  const int gid = blockIdx.x;
  const int bb = gid / H;
  const int ih = gid - bb * H;
  if (bb >= B || ih >= H) return;
  if (T <= 0 || T > 512) return;

  constexpr float kLpFloor = -85.f;
  constexpr float kLamMin = 1e-30f;

  __shared__ float s_lp[512];
  __shared__ float s_acc[512];

  const int t = threadIdx.x;
  float lp_in = 0.f;
  float c_t = 0.f;
  if (t < T) {
    const int64_t ix = idx3(bb, t, ih, sb, st, sh);
    const float lam_t = fmaxf(lam[ix], kLamMin);
    c_t = c[ix];
    lp_in = logf(lam_t);
  }
  s_lp[t] = (t < T) ? lp_in : 0.f;
  __syncthreads();

  inclusive_scan_512(s_lp);
  const float lp = s_lp[t];
  const float lp_eff = fmaxf(lp, kLpFloor);
  const float term = (t < T) ? (c_t * expf(-lp_eff)) : 0.f;
  s_acc[t] = term;
  __syncthreads();

  inclusive_scan_512(s_acc);
  const float acc = s_acc[t];
  if (t < T) {
    const float h0v = h0[bb * h0_sb + ih * h0_sh];
    const float p = expf(lp);
    out[idx3(bb, t, ih, sb, st, sh)] = p * (h0v + acc);
  }
}

__global__ void diagonal_scan_bwd_kernel_v2_512(
    const float* __restrict__ lam,
    const float* __restrict__ h_saved,
    const float* __restrict__ h0,
    const float* __restrict__ grad_out,
    float* __restrict__ d_lam,
    float* __restrict__ d_c,
    float* __restrict__ dh0,
    int B,
    int T,
    int H,
    int64_t sb,
    int64_t st,
    int64_t sh,
    int64_t h0_sb,
    int64_t h0_sh) {
  const int gid = blockIdx.x;
  const int bb = gid / H;
  const int ih = gid - bb * H;
  if (bb >= B || ih >= H) return;
  if (T <= 0 || T > 512) return;

  constexpr float kLpFloor = -85.f;
  constexpr float kLamMin = 1e-30f;

  __shared__ float s_lp[512];
  __shared__ float s_g[512];

  // reversed index r: r=0 corresponds to t=T-1
  const int r = threadIdx.x;
  const int t = T - 1 - r;

  float lp_in = 0.f;
  float go = 0.f;
  if (r < T) {
    const int64_t ix = idx3(bb, t, ih, sb, st, sh);
    const float lam_t = fmaxf(lam[ix], kLamMin);
    go = grad_out[ix];
    lp_in = logf(lam_t);
  }
  s_lp[r] = (r < T) ? lp_in : 0.f;
  __syncthreads();

  inclusive_scan_512(s_lp);
  const float lp = s_lp[r];
  const float lp_eff = fmaxf(lp, kLpFloor);
  const float term = (r < T) ? (go * expf(-lp_eff)) : 0.f;
  s_g[r] = term;
  __syncthreads();

  inclusive_scan_512(s_g);
  if (r < T) {
    const float p = expf(lp);
    const float g_pre = p * s_g[r]; // gradient accumulator at time t
    const int64_t ix = idx3(bb, t, ih, sb, st, sh);
    d_c[ix] = g_pre;
    float h_prev = 0.f;
    if (t > 0) {
      h_prev = h_saved[idx3(bb, t - 1, ih, sb, st, sh)];
    } else {
      h_prev = h0[bb * h0_sb + ih * h0_sh];
    }
    d_lam[ix] = h_prev * g_pre;

    // dh0 uses the post-multiply by lam_0 from the original backward recurrence
    if (t == 0) {
      const float lam0 = lam[ix];
      dh0[bb * h0_sb + ih * h0_sh] = lam0 * g_pre;
    }
  }
}

__global__ void diagonal_scan_fwd_kernel(
    const float* __restrict__ lam,
    const float* __restrict__ c,
    const float* __restrict__ h0,
    float* __restrict__ out,
    int B,
    int T,
    int H,
    int64_t sb,
    int64_t st,
    int64_t sh,
    int64_t h0_sb,
    int64_t h0_sh) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const int bb = gid / H;
  const int ih = gid - bb * H;
  if (bb >= B || ih >= H) {
    return;
  }

  const float h0v = h0[bb * h0_sb + ih * h0_sh];
  float lp = 0.f;
  float acc = 0.f;
  constexpr float kLpFloor = -85.f;
  constexpr float kLamMin = 1e-30f;

  for (int tt = 0; tt < T; ++tt) {
    const int64_t ix = idx3(bb, tt, ih, sb, st, sh);
    const float lam_t = fmaxf(lam[ix], kLamMin);
    const float c_t = c[ix];
    lp += logf(lam_t);
    const float lp_eff = fmaxf(lp, kLpFloor);
    const float inv_p = expf(-lp_eff);
    acc += c_t * inv_p;
    const float p = expf(lp);
    const float h_t = p * h0v + p * acc;
    out[ix] = h_t;
  }
}

__global__ void diagonal_scan_bwd_kernel(
    const float* __restrict__ lam,
    const float* __restrict__ h_saved,
    const float* __restrict__ h0,
    const float* __restrict__ grad_out,
    float* __restrict__ d_lam,
    float* __restrict__ d_c,
    float* __restrict__ dh0,
    int B,
    int T,
    int H,
    int64_t sb,
    int64_t st,
    int64_t sh,
    int64_t h0_sb,
    int64_t h0_sh) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const int bb = gid / H;
  const int ih = gid - bb * H;
  if (bb >= B || ih >= H) {
    return;
  }

  const float h0v = h0[bb * h0_sb + ih * h0_sh];
  float g = 0.f;

  for (int k = 0; k < T; ++k) {
    const int t = T - 1 - k;
    const int64_t ix = idx3(bb, t, ih, sb, st, sh);
    const float go = grad_out[ix];
    const float lam_t = lam[ix];
    g = g + go;
    const int64_t ixm = idx3(bb, t - 1, ih, sb, st, sh);
    const float h_prev = (t > 0) ? h_saved[ixm] : h0v;
    d_c[ix] = g;
    d_lam[ix] = h_prev * g;
    g = lam_t * g;
  }
  dh0[bb * h0_sb + ih * h0_sh] = g;
}

// FP64: align with diagonal_conv_rnn._diagonal_linear_scan_segment (finfo(tiny), lp_eff = max(lp, log(tiny))).
__global__ void diagonal_scan_fwd_kernel_fp64(
    const double* __restrict__ lam,
    const double* __restrict__ c,
    const double* __restrict__ h0,
    double* __restrict__ out,
    int B,
    int T,
    int H,
    int64_t sb,
    int64_t st,
    int64_t sh,
    int64_t h0_sb,
    int64_t h0_sh) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const int bb = gid / H;
  const int ih = gid - bb * H;
  if (bb >= B || ih >= H) {
    return;
  }

  constexpr double kTiny = 2.22507385850720138309023271733240406e-308;
  const double log_tiny = log(kTiny);

  const double h0v = h0[bb * h0_sb + ih * h0_sh];
  double lp = 0.;
  double acc = 0.;

  for (int tt = 0; tt < T; ++tt) {
    const int64_t ix = idx3(bb, tt, ih, sb, st, sh);
    const double lam_t = fmax(lam[ix], kTiny);
    const double c_t = c[ix];
    lp += log(lam_t);
    const double lp_eff = fmax(lp, log_tiny);
    const double inv_p = exp(-lp_eff);
    acc += c_t * inv_p;
    const double p = exp(lp);
    const double h_t = p * h0v + p * acc;
    out[ix] = h_t;
  }
}

__global__ void diagonal_scan_bwd_kernel_fp64(
    const double* __restrict__ lam,
    const double* __restrict__ h_saved,
    const double* __restrict__ h0,
    const double* __restrict__ grad_out,
    double* __restrict__ d_lam,
    double* __restrict__ d_c,
    double* __restrict__ dh0,
    int B,
    int T,
    int H,
    int64_t sb,
    int64_t st,
    int64_t sh,
    int64_t h0_sb,
    int64_t h0_sh) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const int bb = gid / H;
  const int ih = gid - bb * H;
  if (bb >= B || ih >= H) {
    return;
  }

  const double h0v = h0[bb * h0_sb + ih * h0_sh];
  double g = 0.;

  for (int k = 0; k < T; ++k) {
    const int t = T - 1 - k;
    const int64_t ix = idx3(bb, t, ih, sb, st, sh);
    const double go = grad_out[ix];
    const double lam_t = lam[ix];
    g = g + go;
    const int64_t ixm = idx3(bb, t - 1, ih, sb, st, sh);
    const double h_prev = (t > 0) ? h_saved[ixm] : h0v;
    d_c[ix] = g;
    d_lam[ix] = h_prev * g;
    g = lam_t * g;
  }
  dh0[bb * h0_sb + ih * h0_sh] = g;
}

} // namespace

torch::Tensor diag_scan_fwd_cuda(torch::Tensor lam, torch::Tensor c, torch::Tensor h0) {
  TORCH_CHECK(lam.is_cuda(), "diag_scan_fwd_cuda: CUDA tensors required");
  TORCH_CHECK(lam.dim() == 3 && c.dim() == 3 && h0.dim() == 2, "lam,c must be (B,T,H), h0 (B,H)");
  TORCH_CHECK(lam.scalar_type() == torch::kFloat32, "diag_scan_fwd_cuda: float32 only");
  TORCH_CHECK(c.scalar_type() == torch::kFloat32 && h0.scalar_type() == torch::kFloat32,
 "diag_scan_fwd_cuda: float32 only");
  lam = lam.contiguous();
  c = c.contiguous();
  h0 = h0.contiguous();
  const int B = static_cast<int>(lam.size(0));
  const int T = static_cast<int>(lam.size(1));
  const int H = static_cast<int>(lam.size(2));
  TORCH_CHECK(c.sizes() == lam.sizes(), "c shape must match lam");
  TORCH_CHECK(h0.sizes() == torch::IntArrayRef({B, H}), "h0 shape must be (B,H)");

  auto out = torch::empty({B, T, H}, lam.options());
  if (B == 0 || H == 0 || T == 0) {
    return out;
  }

  const auto s = lam.strides();
  TORCH_CHECK(s[2] == 1, "lam must be contiguous in H (stride 1)");
  const int64_t sb = s[0];
  const int64_t st = s[1];
  const int64_t sh = s[2];
  const auto hs = h0.strides();
  TORCH_CHECK(hs[1] == 1, "h0 must be contiguous in H");

  constexpr int kThreads = 256;
  const int64_t total = static_cast<int64_t>(B) * static_cast<int64_t>(H);
  const int nblocks = static_cast<int>((total + kThreads - 1) / kThreads);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(lam.device().index());
  diagonal_scan_fwd_kernel<<<nblocks, kThreads, 0, stream>>>(
      lam.data_ptr<float>(),
      c.data_ptr<float>(),
      h0.data_ptr<float>(),
      out.data_ptr<float>(),
      B,
      T,
      H,
      sb,
      st,
      sh,
      hs[0],
      hs[1]);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor diag_scan_fwd_v2_cuda(torch::Tensor lam, torch::Tensor c, torch::Tensor h0) {
  TORCH_CHECK(lam.is_cuda(), "diag_scan_fwd_v2_cuda: CUDA tensors required");
  TORCH_CHECK(lam.dim() == 3 && c.dim() == 3 && h0.dim() == 2, "lam,c must be (B,T,H), h0 (B,H)");
  TORCH_CHECK(lam.scalar_type() == torch::kFloat32, "diag_scan_fwd_v2_cuda: float32 only");
  TORCH_CHECK(c.scalar_type() == torch::kFloat32 && h0.scalar_type() == torch::kFloat32,
              "diag_scan_fwd_v2_cuda: float32 only");
  lam = lam.contiguous();
  c = c.contiguous();
  h0 = h0.contiguous();
  const int B = static_cast<int>(lam.size(0));
  const int T = static_cast<int>(lam.size(1));
  const int H = static_cast<int>(lam.size(2));
  TORCH_CHECK(c.sizes() == lam.sizes(), "c shape must match lam");
  TORCH_CHECK(h0.sizes() == torch::IntArrayRef({B, H}), "h0 shape must be (B,H)");

  auto out = torch::empty({B, T, H}, lam.options());
  if (B == 0 || H == 0 || T == 0) {
    return out;
  }

  const auto s = lam.strides();
  TORCH_CHECK(s[2] == 1, "lam must be contiguous in H (stride 1)");
  const int64_t sb = s[0];
  const int64_t st = s[1];
  const int64_t sh = s[2];
  const auto hs = h0.strides();
  TORCH_CHECK(hs[1] == 1, "h0 must be contiguous in H");

  if (T <= 512) {
    const int nblocks = B * H;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(lam.device().index());
    diagonal_scan_fwd_kernel_v2_512<<<nblocks, 512, 0, stream>>>(
        lam.data_ptr<float>(),
        c.data_ptr<float>(),
        h0.data_ptr<float>(),
        out.data_ptr<float>(),
        B,
        T,
        H,
        sb,
        st,
        sh,
        hs[0],
        hs[1]);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
  }
  // Fallback to v1 when T is large.
  return diag_scan_fwd_cuda(lam, c, h0);
}

std::vector<torch::Tensor> diag_scan_bwd_cuda(
    torch::Tensor lam,
    torch::Tensor h_saved,
    torch::Tensor h0,
    torch::Tensor grad_out) {
  TORCH_CHECK(lam.is_cuda(), "diag_scan_bwd_cuda: CUDA tensors required");
  TORCH_CHECK(
      lam.scalar_type() == torch::kFloat32 && h_saved.scalar_type() == torch::kFloat32 &&
 h0.scalar_type() == torch::kFloat32 && grad_out.scalar_type() == torch::kFloat32,
      "diag_scan_bwd_cuda: float32 only");
  lam = lam.contiguous();
  h_saved = h_saved.contiguous();
  h0 = h0.contiguous();
  grad_out = grad_out.contiguous();

  const int B = static_cast<int>(lam.size(0));
  const int T = static_cast<int>(lam.size(1));
  const int H = static_cast<int>(lam.size(2));
  TORCH_CHECK(h_saved.sizes() == lam.sizes() && grad_out.sizes() == lam.sizes(), "shape mismatch");
  TORCH_CHECK(h0.sizes() == torch::IntArrayRef({B, H}), "h0 (B,H)");

  auto d_lam = torch::zeros_like(lam);
  auto d_c = torch::zeros_like(lam);
  auto dh0 = torch::zeros_like(h0);

  if (B == 0 || H == 0 || T == 0) {
    return {d_lam, d_c, dh0};
  }

  const auto s = lam.strides();
  TORCH_CHECK(s[2] == 1, "lam must be contiguous in H");
  const int64_t sb = s[0];
  const int64_t st = s[1];
  const int64_t sh = s[2];
  const auto hs = h0.strides();
  TORCH_CHECK(hs[1] == 1, "h0 must be contiguous in H");

  constexpr int kThreads = 256;
  const int64_t total = static_cast<int64_t>(B) * static_cast<int64_t>(H);
  const int nblocks = static_cast<int>((total + kThreads - 1) / kThreads);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(lam.device().index());
  diagonal_scan_bwd_kernel<<<nblocks, kThreads, 0, stream>>>(
      lam.data_ptr<float>(),
      h_saved.data_ptr<float>(),
      h0.data_ptr<float>(),
      grad_out.data_ptr<float>(),
      d_lam.data_ptr<float>(),
      d_c.data_ptr<float>(),
      dh0.data_ptr<float>(),
      B,
      T,
      H,
      sb,
      st,
      sh,
      hs[0],
      hs[1]);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {d_lam, d_c, dh0};
}

std::vector<torch::Tensor> diag_scan_bwd_v2_cuda(
    torch::Tensor lam,
    torch::Tensor h_saved,
    torch::Tensor h0,
    torch::Tensor grad_out) {
  TORCH_CHECK(lam.is_cuda(), "diag_scan_bwd_v2_cuda: CUDA tensors required");
  TORCH_CHECK(
      lam.scalar_type() == torch::kFloat32 && h_saved.scalar_type() == torch::kFloat32 &&
          h0.scalar_type() == torch::kFloat32 && grad_out.scalar_type() == torch::kFloat32,
      "diag_scan_bwd_v2_cuda: float32 only");
  lam = lam.contiguous();
  h_saved = h_saved.contiguous();
  h0 = h0.contiguous();
  grad_out = grad_out.contiguous();

  const int B = static_cast<int>(lam.size(0));
  const int T = static_cast<int>(lam.size(1));
  const int H = static_cast<int>(lam.size(2));
  TORCH_CHECK(h_saved.sizes() == lam.sizes() && grad_out.sizes() == lam.sizes(), "shape mismatch");
  TORCH_CHECK(h0.sizes() == torch::IntArrayRef({B, H}), "h0 (B,H)");

  auto d_lam = torch::zeros_like(lam);
  auto d_c = torch::zeros_like(lam);
  auto dh0 = torch::zeros_like(h0);
  if (B == 0 || H == 0 || T == 0) {
    return {d_lam, d_c, dh0};
  }

  const auto s = lam.strides();
  TORCH_CHECK(s[2] == 1, "lam must be contiguous in H (stride 1)");
  const int64_t sb = s[0];
  const int64_t st = s[1];
  const int64_t sh = s[2];
  const auto hs = h0.strides();
  TORCH_CHECK(hs[1] == 1, "h0 must be contiguous in H");

  if (T <= 512) {
    const int nblocks = B * H;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(lam.device().index());
    diagonal_scan_bwd_kernel_v2_512<<<nblocks, 512, 0, stream>>>(
        lam.data_ptr<float>(),
        h_saved.data_ptr<float>(),
        h0.data_ptr<float>(),
        grad_out.data_ptr<float>(),
        d_lam.data_ptr<float>(),
        d_c.data_ptr<float>(),
        dh0.data_ptr<float>(),
        B,
        T,
        H,
        sb,
        st,
        sh,
        hs[0],
        hs[1]);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {d_lam, d_c, dh0};
  }
  return diag_scan_bwd_cuda(lam, h_saved, h0, grad_out);
}

torch::Tensor diag_scan_fwd_fp64_cuda(torch::Tensor lam, torch::Tensor c, torch::Tensor h0) {
  TORCH_CHECK(lam.is_cuda(), "diag_scan_fwd_fp64_cuda: CUDA tensors required");
  TORCH_CHECK(lam.dim() == 3 && c.dim() == 3 && h0.dim() == 2, "lam,c must be (B,T,H), h0 (B,H)");
  TORCH_CHECK(
      lam.scalar_type() == torch::kFloat64 && c.scalar_type() == torch::kFloat64 &&
          h0.scalar_type() == torch::kFloat64,
      "diag_scan_fwd_fp64_cuda: float64 only");
  lam = lam.contiguous();
  c = c.contiguous();
  h0 = h0.contiguous();
  const int B = static_cast<int>(lam.size(0));
  const int T = static_cast<int>(lam.size(1));
  const int H = static_cast<int>(lam.size(2));
  TORCH_CHECK(c.sizes() == lam.sizes(), "c shape must match lam");
  TORCH_CHECK(h0.sizes() == torch::IntArrayRef({B, H}), "h0 shape must be (B,H)");

  auto out = torch::empty({B, T, H}, lam.options());
  if (B == 0 || H == 0 || T == 0) {
    return out;
  }

  const auto s = lam.strides();
  TORCH_CHECK(s[2] == 1, "lam must be contiguous in H (stride 1)");
  const int64_t sb = s[0];
  const int64_t st = s[1];
  const int64_t sh = s[2];
  const auto hs = h0.strides();
  TORCH_CHECK(hs[1] == 1, "h0 must be contiguous in H");

  constexpr int kThreads = 256;
  const int64_t total = static_cast<int64_t>(B) * static_cast<int64_t>(H);
  const int nblocks = static_cast<int>((total + kThreads - 1) / kThreads);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(lam.device().index());
  diagonal_scan_fwd_kernel_fp64<<<nblocks, kThreads, 0, stream>>>(
      lam.data_ptr<double>(),
      c.data_ptr<double>(),
      h0.data_ptr<double>(),
      out.data_ptr<double>(),
      B,
      T,
      H,
      sb,
      st,
      sh,
      hs[0],
      hs[1]);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

std::vector<torch::Tensor> diag_scan_bwd_fp64_cuda(
    torch::Tensor lam,
    torch::Tensor h_saved,
    torch::Tensor h0,
    torch::Tensor grad_out) {
  TORCH_CHECK(lam.is_cuda(), "diag_scan_bwd_fp64_cuda: CUDA tensors required");
  TORCH_CHECK(
      lam.scalar_type() == torch::kFloat64 && h_saved.scalar_type() == torch::kFloat64 &&
          h0.scalar_type() == torch::kFloat64 && grad_out.scalar_type() == torch::kFloat64,
      "diag_scan_bwd_fp64_cuda: float64 only");
  lam = lam.contiguous();
  h_saved = h_saved.contiguous();
  h0 = h0.contiguous();
  grad_out = grad_out.contiguous();

  const int B = static_cast<int>(lam.size(0));
  const int T = static_cast<int>(lam.size(1));
  const int H = static_cast<int>(lam.size(2));
  TORCH_CHECK(h_saved.sizes() == lam.sizes() && grad_out.sizes() == lam.sizes(), "shape mismatch");
  TORCH_CHECK(h0.sizes() == torch::IntArrayRef({B, H}), "h0 (B,H)");

  auto d_lam = torch::zeros_like(lam);
  auto d_c = torch::zeros_like(lam);
  auto dh0 = torch::zeros_like(h0);

  if (B == 0 || H == 0 || T == 0) {
    return {d_lam, d_c, dh0};
  }

  const auto s = lam.strides();
  TORCH_CHECK(s[2] == 1, "lam must be contiguous in H");
  const int64_t sb = s[0];
  const int64_t st = s[1];
  const int64_t sh = s[2];
  const auto hs = h0.strides();
  TORCH_CHECK(hs[1] == 1, "h0 must be contiguous in H");

  constexpr int kThreads = 256;
  const int64_t total = static_cast<int64_t>(B) * static_cast<int64_t>(H);
  const int nblocks = static_cast<int>((total + kThreads - 1) / kThreads);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(lam.device().index());
  diagonal_scan_bwd_kernel_fp64<<<nblocks, kThreads, 0, stream>>>(
      lam.data_ptr<double>(),
      h_saved.data_ptr<double>(),
      h0.data_ptr<double>(),
      grad_out.data_ptr<double>(),
      d_lam.data_ptr<double>(),
      d_c.data_ptr<double>(),
      dh0.data_ptr<double>(),
      B,
      T,
      H,
      sb,
      st,
      sh,
      hs[0],
      hs[1]);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {d_lam, d_c, dh0};
}
