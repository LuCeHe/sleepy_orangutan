"""
Diagonal **linear** recurrence with input-dependent gains:

``h_t = λ_t ⊙ h_{t-1} + c_t``, ``c_t = W x_t + b``. Default ``λ_t = exp(-softplus(causal_conv(x)_t))`` in ``(0, 1]``; ``lam_gate="sigmoid"`` uses
``λ_t = σ(causal_conv(x)_t)``.

This is the affine associative scan along time (elementwise across hidden dims). Closed form:

``h_t = (∏_{k=1}^{t} λ_k) ⊙ h_0 + Σ_{k=1}^{t} (∏_{j=k+1}^{t} λ_j) ⊙ c_k``

Vectorized form: with suffix products ``R_m = ∏_{j=m}^{T} λ_j`` (append ``λ_{T+1}=1``),
``h_t = (∏_{i=0}^{t} λ_i) ⊙ h_0 + (cumsum_k(c_k ⊙ R_{k+1}))_t / R_{t+1}`` (indices 0-based).

**Products** use ``exp(cumsum(log λ))`` instead of ``cumprod(λ)`` so long runs of small ``λ`` avoid
underflow; the ``c`` branch still mixes in linear space (a sum of terms, not a pure product).

For ``float16`` / ``bfloat16`` / ``float32`` inputs, the scan runs in **float64** internally and casts
back. With ``ℓ_t = cumsum_t(log λ)`` and ``P_t = exp(ℓ_t)`` (unclamped for the outer factor),

``h_t = P_t h_0 + Σ_{s≤t} c_s * exp(ℓ_t - ℓ_s)``. In **O(L)** we use ``ℓ^{eff}_t = max(ℓ_t, log ε)``
with ``ε =`` machine ``tiny`` in compute dtype so ``c_s / max(P_s, ε) = c_s * exp(-ℓ^{eff}_s)`` stays
finite; the outer ``P_t`` is still ``exp(ℓ_t)``. Without that floor, ``exp(-ℓ)`` blows up when the
prefix underflows and the forward **cumsum** fills with ``NaN`` (especially for full-length ``T``).
Chunked carries match a single full-length pass (older ``partial / R`` suffix code did not).

**Chunking** (optional ``chunk_size``): run the same closed form on time segments
``[t_0:t_1), [t_1:t_2), …`` carrying ``h_{t_k}`` across boundaries. Exact same recurrence, same
FLOPs. **Gain:** sometimes lower **peak** memory for scan intermediates (they scale with segment
length, not ``T``). **Cost:** more kernel launches; on GPU a single long pass is often **faster**.
``DiagonalCausalConvRNN`` defaults to **one full-length pass** (``scan_chunk_size=None``). Pass an
integer (e.g. ``64``) if you want time chunking for memory or to match Gated DeltaNet block size.

**Segment reduce** (``scan_segment_reduce``):

- ``"scan"`` (default): O(L) closed form per segment (prefix cumsum form above).
- ``"causal_bmm"``: build the causal lower-triangular matrix
  ``M_{t,s} = (∏_{i=0}^{t} λ_i) / (∏_{i=0}^{s} λ_i)`` for ``t ≥ s`` and compute
  ``h = M @ c + (\\prod_{i=0}^{t} \\lambda_i) * h_0`` (elementwise over hidden dims). Same math,
  but **O(L²)** FLOPs and **O(H·L²)** memory for ``M``. Use only for **small** segment length ``L``
  (e.g. ``scan_chunk_size=64``); for one segment with ``L = T`` large, use ``"scan"``.
"""

from __future__ import annotations

import math
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from thepebbletrail_official.neuron_utils._more_layers.some_convs import get_conv1d

SegmentReduce = Literal["scan", "causal_bmm"]
LamGate = Literal["sigmoid", "softplus_exp"]


def _align_conv_time(y: torch.Tensor, target_len: int) -> torch.Tensor:
    """Crop or error so conv output matches input length ``(B, T, *)`` for the scan."""
    if y.shape[1] == target_len:
        return y
    if y.shape[1] > target_len:
        return y[:, :target_len, :]
    raise ValueError(
        f"diag_conv output time {y.shape[1]} < input time {target_len}; choose another conv_type.",
    )


def diagonal_prep_scan_inputs(module: DiagonalCausalConvRNN, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``lam``, ``c`` passed to :func:`diagonal_linear_scan` (same as :meth:`DiagonalCausalConvRNN.forward`)."""
    z = module.diag_conv(x)
    z = _align_conv_time(z, x.shape[1])
    if module.lam_gate == "softplus_exp":
        lam = torch.exp(-F.softplus(z))
    else:
        lam = torch.sigmoid(z)
    c = F.linear(x, module.weight_in) + module.bias
    return lam, c


def _diagonal_linear_scan_segment(
    lam: torch.Tensor,
    c: torch.Tensor,
    h0: torch.Tensor,
) -> torch.Tensor:
    """One segment of length L: ``h_t = lam_t * h_{t-1} + c_t``, ``lam,c``: (B, L, H), ``h0``: (B, H)."""
    b, ell, hdim = lam.shape
    if c.shape != lam.shape:
        raise ValueError(f"c shape {c.shape} must match lam {lam.shape}")
    device, dtype = lam.device, lam.dtype
    # float32: use f64 for prefix products and c/P ratios on long segments.
    compute_dtype = torch.float64 if dtype in (torch.float16, torch.bfloat16, torch.float32) else dtype
    tiny = torch.finfo(compute_dtype).tiny
    lam_c = lam.to(compute_dtype)
    c_c = c.to(compute_dtype)
    h0_c = h0.to(compute_dtype)

    lam_safe = torch.clamp(lam_c, min=tiny)
    log_lam = torch.log(lam_safe)
    log_prefix = torch.cumsum(log_lam, dim=1)
    prefix = torch.exp(log_prefix)
    # Denominator uses safe_p = max(P, ε): ℓ_eff = max(ℓ, log ε); c/safe_p = c * exp(-ℓ_eff).
    # Outer P_t = exp(ℓ_t) unchanged (matches segment carry across chunk boundaries).
    log_tiny = torch.log(torch.tensor(tiny, dtype=compute_dtype, device=device))
    lp_eff = torch.maximum(log_prefix, log_tiny)
    acc = torch.cumsum(c_c * torch.exp(-lp_eff), dim=1)
    out = prefix * h0_c.unsqueeze(1) + prefix * acc
    return out.to(dtype)


def _diagonal_linear_scan_segment_causal_bmm(
    lam: torch.Tensor,
    c: torch.Tensor,
    h0: torch.Tensor,
) -> torch.Tensor:
    """Same recurrence as :func:`_diagonal_linear_scan_segment`, via lower-triangular ``M`` and batched ``bmm``.

    ``h_t = P_t * h_0 + sum_{s<=t} (P_t / P_s) * c_s`` with ``P_t = prod_{i=0}^t lam_i`` (0-based).
    Complexity O(L^2 * H) in time and O(H * L^2) for the materialized mask path.
    """
    b, ell, hdim = lam.shape
    if c.shape != lam.shape:
        raise ValueError(f"c shape {c.shape} must match lam {lam.shape}")
    device, dtype = lam.device, lam.dtype
    compute_dtype = torch.float64 if dtype in (torch.float16, torch.bfloat16, torch.float32) else dtype
    tiny = torch.finfo(compute_dtype).tiny
    lam_c = lam.to(compute_dtype)
    c_c = c.to(compute_dtype)
    h0_c = h0.to(compute_dtype)

    lam_safe = torch.clamp(lam_c, min=tiny)
    log_lam = torch.log(lam_safe)
    log_prefix = torch.cumsum(log_lam, dim=1)
    # (B, H, L)
    lp = log_prefix.transpose(1, 2).contiguous()
    # M[b,h,t,s] = exp(lp[b,h,t] - lp[b,h,s]) on lower triangle (causal)
    diff = lp.unsqueeze(-1) - lp.unsqueeze(-2)
    tril = torch.tril(torch.ones(ell, ell, device=device, dtype=torch.bool))
    diff = diff.masked_fill(~tril.view(1, 1, ell, ell), float("-inf"))
    m = torch.exp(diff)
    # (B, H, L, L) @ (B, H, L, 1) -> (B, H, L)
    c_bt = c_c.transpose(1, 2).unsqueeze(-1)
    part = torch.matmul(m, c_bt).squeeze(-1).transpose(1, 2)
    prefix = torch.exp(log_prefix)
    out = part + prefix * h0_c.unsqueeze(1)
    return out.to(dtype)


def diagonal_linear_scan(
    lam: torch.Tensor,
    c: torch.Tensor,
    h0: Optional[torch.Tensor] = None,
    chunk_size: Optional[int] = None,
    *,
    segment_reduce: SegmentReduce = "scan",
    use_triton_scan: Optional[bool] = None,
) -> torch.Tensor:
    """
    Compute ``h_t = lam_t * h_{t-1} + c_t`` for all ``t`` (elementwise ``*``).

    Args:
        lam: (B, T, H) strictly in ``(0, 1]`` (e.g. from ``σ`` or ``exp(-softplus(·))``).
        c: (B, T, H)
        h0: (B, H) or None for zeros.
        chunk_size: if set, process time in chunks of this length (carry ``h`` across chunks).
        segment_reduce: ``"scan"`` (O(L) per segment) or ``"causal_bmm"`` (O(L^2); use small ``L``).
        use_triton_scan: if ``True``, try CUDA Triton scan without env; if ``False``, never Triton;
            if ``None`` (default), use env ``THEPEBBLETRAIL_DIAG_SCAN_TRITON=1`` (``segment_reduce="scan"`` only).
            Triton uses fp32 internally vs PyTorch fp64 scan; see benchmark ``diag triton`` row.

    Optional CUDA Triton path (``segment_reduce="scan"`` only): env ``THEPEBBLETRAIL_DIAG_SCAN_TRITON=1``
    or ``use_triton_scan=True``. Without grad, uses a bare forward kernel; with grad on ``lam``/``c``/``h0``,
    uses :func:`diagonal_linear_scan_triton` (custom autograd, fp32).

    Returns:
        h: (B, T, H)
    """
    b, t, hdim = lam.shape
    if c.shape != lam.shape:
        raise ValueError(f"c shape {c.shape} must match lam {lam.shape}")
    if segment_reduce not in ("scan", "causal_bmm"):
        raise ValueError(f"segment_reduce must be 'scan' or 'causal_bmm', got {segment_reduce!r}")
    device, dtype = lam.device, lam.dtype
    if h0 is None:
        h0 = torch.zeros(b, hdim, device=device, dtype=dtype)
    else:
        h0 = h0.contiguous()

    seg = _diagonal_linear_scan_segment_causal_bmm if segment_reduce == "causal_bmm" else _diagonal_linear_scan_segment

    maybe_triton = None
    if segment_reduce == "scan" and use_triton_scan is not False:
        try:
            from sleepy_orangutan.gateddeltanet.diagonal_scan_triton import (
                maybe_diagonal_linear_scan_triton,
            )

            maybe_triton = maybe_diagonal_linear_scan_triton
        except ModuleNotFoundError:
            try:
                from diagonal_scan_triton import maybe_diagonal_linear_scan_triton

                maybe_triton = maybe_diagonal_linear_scan_triton
            except ModuleNotFoundError:
                maybe_triton = None

    def _try_triton(la: torch.Tensor, cc: torch.Tensor, h0_: torch.Tensor) -> Optional[torch.Tensor]:
        if maybe_triton is None:
            return None
        if use_triton_scan is True:
            return maybe_triton(la, cc, h0_, force=True)
        return maybe_triton(la, cc, h0_)

    if chunk_size is None or t <= chunk_size:
        ht = _try_triton(lam, c, h0)
        if ht is not None:
            return ht
        return seg(lam, c, h0)

    out_parts: list[torch.Tensor] = []
    state = h0
    for s in range(0, t, chunk_size):
        e = min(s + chunk_size, t)
        seg_tensor = _try_triton(lam[:, s:e], c[:, s:e], state)
        if seg_tensor is None:
            seg_tensor = seg(lam[:, s:e], c[:, s:e], state)
        out_parts.append(seg_tensor)
        state = seg_tensor[:, -1, :].contiguous()
    return torch.cat(out_parts, dim=1)


class DiagonalCausalConvRNN(nn.Module):
    """
    ``h_t = λ_t ⊙ h_{t-1} + (W x_t + b)`` with ``z = causal_conv(x)``. Default ``λ = exp(-softplus(z))``;
    ``lam_gate="sigmoid"`` uses ``λ = σ(z)``.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        kernel_size_t=2,
        kernel_size_w=16,
        conv_type: str = "cudaext2dv2br",
        scan_chunk_size: Optional[int] = None,
        scan_segment_reduce: SegmentReduce = "scan",
        lam_gate: LamGate = "softplus_exp",
        use_triton_scan: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size_t = kernel_size_t
        self.kernel_size_w = kernel_size_w
        self.conv_type = conv_type
        self.scan_chunk_size = scan_chunk_size
        self.scan_segment_reduce = scan_segment_reduce
        self.lam_gate = lam_gate
        self.use_triton_scan = use_triton_scan
        self.diag_conv = get_conv1d(
            in_channels=input_size,
            out_channels=hidden_size,
            conv_type=conv_type,
            kernel_size_t=kernel_size_t,
            kernel_size_w=kernel_size_w,
            bias=True,
            dilation=1,
            notes="",
        )
        self.weight_in = nn.Parameter(torch.empty(hidden_size, input_size))
        self.bias = nn.Parameter(torch.empty(hidden_size))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        nn.init.uniform_(self.weight_in, -stdv, stdv)
        nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        scan_chunk_size: Optional[int] = None,
        scan_segment_reduce: Optional[SegmentReduce] = None,
        use_triton_scan: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, input_size)
            h0: (B, hidden_size) optional
            scan_chunk_size: segment length for this forward. If ``None``, uses ``self.scan_chunk_size``
                (constructor default is one full-length pass: ``None``). Pass a positive int to chunk time;
                use ``scan_chunk_size >= T`` for an explicit single segment without changing the module default.
            scan_segment_reduce: optional override for ``self.scan_segment_reduce`` (``"scan"`` vs
                ``"causal_bmm"``).
            use_triton_scan: if not ``None``, overrides ``self.use_triton_scan`` for this call.
        Returns:
            outputs (B, T, hidden_size), h_T (B, hidden_size)
        """
        lam, c = diagonal_prep_scan_inputs(self, x)
        cs = self.scan_chunk_size if scan_chunk_size is None else scan_chunk_size
        sr = self.scan_segment_reduce if scan_segment_reduce is None else scan_segment_reduce
        ut = self.use_triton_scan if use_triton_scan is None else use_triton_scan
        h = diagonal_linear_scan(
            lam, c, h0=h0, chunk_size=cs, segment_reduce=sr, use_triton_scan=ut
        )
        return h, h[:, -1, :].contiguous()


def _causality_suffix_invariance() -> None:
    """Changing x[:, cut:] must not change h[:, :cut] (strict left-to-right dependence)."""
    torch.manual_seed(0)
    b, t, d, hdim, k = 2, 48, 8, 16, 4
    cut = 17
    m = DiagonalCausalConvRNN(d, hdim, kernel_size=k)
    m.eval()
    x = torch.randn(b, t, d)
    x_alt = x.clone()
    x_alt[:, cut:, :] = torch.randn_like(x_alt[:, cut:, :])
    with torch.no_grad():
        h, _ = m(x)
        h_alt, _ = m(x_alt)
        
    torch.testing.assert_close(h[:, :cut, :], h_alt[:, :cut, :], rtol=1e-4, atol=1e-5)


def _chunk_matches_full_scan() -> None:
    torch.manual_seed(1)
    b, t, hdim = 3, 100, 32
    lam = torch.sigmoid(torch.randn(b, t, hdim))
    c = torch.randn(b, t, hdim)
    h0 = torch.randn(b, hdim)
    h_full = diagonal_linear_scan(lam, c, h0=h0, chunk_size=None)
    for cs in (7, 16, 50):
        h_ch = diagonal_linear_scan(lam, c, h0=h0, chunk_size=cs)
        torch.testing.assert_close(h_ch, h_full, rtol=1e-5, atol=1e-5)
        h_bmm = diagonal_linear_scan(lam, c, h0=h0, chunk_size=cs, segment_reduce="causal_bmm")
        torch.testing.assert_close(h_bmm, h_full, rtol=1e-5, atol=1e-5)


def _causal_bmm_matches_scan_on_segment() -> None:
    torch.manual_seed(2)
    b, ell, hdim = 2, 17, 24
    lam = torch.sigmoid(torch.randn(b, ell, hdim))
    c = torch.randn(b, ell, hdim)
    h0 = torch.randn(b, hdim)
    a = _diagonal_linear_scan_segment(lam, c, h0)
    b_ = _diagonal_linear_scan_segment_causal_bmm(lam, c, h0)
    torch.testing.assert_close(a, b_, rtol=1e-5, atol=1e-5)


def _causality_suffix_invariance_sigmoid() -> None:
    torch.manual_seed(0)
    b, t, d, hdim, k = 2, 48, 8, 16, 4
    cut = 17
    m = DiagonalCausalConvRNN(d, hdim, kernel_size=k, lam_gate="sigmoid")
    m.eval()
    x = torch.randn(b, t, d)
    x_alt = x.clone()
    x_alt[:, cut:, :] = torch.randn_like(x_alt[:, cut:, :])
    with torch.no_grad():
        h, _ = m(x)
        h_alt, _ = m(x_alt)
    torch.testing.assert_close(h[:, :cut, :], h_alt[:, :cut, :], rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    _causality_suffix_invariance()
    _causality_suffix_invariance_sigmoid()
    _chunk_matches_full_scan()
    _causal_bmm_matches_scan_on_segment()
    print("tests passed")
