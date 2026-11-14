#include <torch/extension.h>

torch::Tensor diag_scan_fwd_cuda(torch::Tensor lam, torch::Tensor c, torch::Tensor h0);

std::vector<torch::Tensor> diag_scan_bwd_cuda(
    torch::Tensor lam,
    torch::Tensor h_saved,
    torch::Tensor h0,
    torch::Tensor grad_out);

torch::Tensor diag_scan_fwd_v2_cuda(torch::Tensor lam, torch::Tensor c, torch::Tensor h0);

std::vector<torch::Tensor> diag_scan_bwd_v2_cuda(
    torch::Tensor lam,
    torch::Tensor h_saved,
    torch::Tensor h0,
    torch::Tensor grad_out);

torch::Tensor diag_scan_fwd_fp64_cuda(torch::Tensor lam, torch::Tensor c, torch::Tensor h0);

std::vector<torch::Tensor> diag_scan_bwd_fp64_cuda(
    torch::Tensor lam,
    torch::Tensor h_saved,
    torch::Tensor h0,
    torch::Tensor grad_out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("diag_scan_fwd", &diag_scan_fwd_cuda, "Diagonal linear scan forward (CUDA, fp32)");
  m.def("diag_scan_bwd", &diag_scan_bwd_cuda, "Diagonal linear scan backward (CUDA, fp32)");
  m.def("diag_scan_fwd_v2", &diag_scan_fwd_v2_cuda, "Diagonal linear scan forward (CUDA, fp32, v2)");
  m.def("diag_scan_bwd_v2", &diag_scan_bwd_v2_cuda, "Diagonal linear scan backward (CUDA, fp32, v2)");
  m.def("diag_scan_fwd_fp64", &diag_scan_fwd_fp64_cuda, "Diagonal linear scan forward (CUDA, fp64)");
  m.def("diag_scan_bwd_fp64", &diag_scan_bwd_fp64_cuda, "Diagonal linear scan backward (CUDA, fp64)");
}
