"""Build RT/STM trunk layers (vendored StateTensorModel path for VLA students)."""

from __future__ import annotations

import re
from types import SimpleNamespace

from minipi05.stm.str2val import str2val
from minipi05.stm.state_tensor_model import StateTensorModel, default_stm_head_dims


def _delta_type(notes: str) -> str:
    m = re.search(r"delta:([^_:]+)", notes or "")
    return m.group(1).lower() if m else "gateddelta"


def _gdn_num_heads_head_dim(hidden_size: int, notes: str = "") -> tuple[int, int]:
    notes = notes or ""
    forced_hd = str2val(notes, "headdim", default=0, output_type=int)
    forced_nh = str2val(notes, "numheads", default=0, output_type=int)
    if forced_hd > 0 and forced_nh > 0:
        return forced_nh, forced_hd
    key_dim = max(1, int(round(0.75 * hidden_size)))
    if forced_nh > 0 and key_dim % forced_nh == 0:
        return forced_nh, key_dim // forced_nh
    if forced_hd > 0:
        nh = forced_nh if forced_nh > 0 else max(1, hidden_size // forced_hd)
        return nh, forced_hd
    for num_heads in (6, 8, 4, 2, 1):
        if key_dim % num_heads == 0:
            return num_heads, key_dim // num_heads
    return 1, key_dim


def _build_stm_layer(config: SimpleNamespace, *, layer_idx: int = 0) -> StateTensorModel:
    _ = layer_idx
    stm_notes = getattr(config, "notes", "") or ""
    _nh, _hd = _gdn_num_heads_head_dim(int(config.hidden_size), stm_notes)
    stm_expand_v = str2val(stm_notes, "stmexpandv", default=2.0, output_type=float)
    dk_def, du_def, dv_def = default_stm_head_dims(
        int(config.hidden_size), _nh, expand_v=stm_expand_v, head_dim=_hd,
    )
    dk = str2val(stm_notes, "stmdk", default=dk_def, output_type=int)
    use_matrix_state = "stmmatrixstate" in stm_notes or "stm_matrixstate" in stm_notes
    du = 1 if use_matrix_state else str2val(stm_notes, "stmdu", default=du_def, output_type=int)
    dv = str2val(stm_notes, "stmdv", default=dv_def, output_type=int)
    use_exp2 = "stmexp2" in stm_notes
    use_triton = "stmnotriton" not in stm_notes
    train_pytorch_core_only = "stmtorchonly" in stm_notes
    stm_torch_chunk = str2val(stm_notes, "stmtorchchunk", default=0, output_type=int)
    if "stmanoautotune" in stm_notes and int(stm_torch_chunk) <= 0:
        stm_torch_chunk = 64
    stm_bwd_default = "chunked" if int(stm_torch_chunk) > 0 else "dense"
    stm_bwd = str2val(stm_notes, "stmbwd", default=stm_bwd_default, output_type=str).lower()
    gdn_conv_size = str2val(stm_notes, "stmgdnconv", default=4, output_type=int)
    stm_gate_type = str2val(stm_notes, "stmgatetype", default="sigmoid", output_type=str)
    use_gla_gate = "stmglagate" in stm_notes
    gate_logit_normalizer = str2val(stm_notes, "stmgatenorm", default=16, output_type=int)
    gate_low_rank_dim = str2val(stm_notes, "stmgklowrank", default=16, output_type=int)
    gate_clamp_min = str2val(stm_notes, "stmgateclamp", default=None, output_type=float)
    gate_cumulative = "stmgcumsum" in stm_notes or "stmvitgcumsum" in stm_notes
    use_qk_l2norm_in_kernel = "stmnotnormqk" not in stm_notes
    use_output_gate = "stmnogout" not in stm_notes
    fuse_output_gate = "stmgouttime" in stm_notes
    use_write_gate = "stmnobeta" not in stm_notes
    fuse_write_gate = "ccbeta" in stm_notes
    use_two_betas2 = "stmtwobetas2" in stm_notes
    use_two_betas = use_two_betas2 or "stmtwobetas" in stm_notes
    allow_neg_eigval_write = "stmbetadouble" in stm_notes
    use_delta3 = "stmdelta3" in stm_notes or "stm_delta3" in stm_notes
    use_delta2 = not use_delta3 and ("stmdelta2" in stm_notes or "stm_delta2" in stm_notes)
    use_fla_fused = "stmfused" in stm_notes
    use_fla_matrix = use_fla_fused or "stmfla" in stm_notes
    use_delta = (
        not use_delta3
        and not use_delta2
        and ("stmdelta" in stm_notes or "stm_delta" in stm_notes)
    )
    attn_scale = "pstmattnscale" in stm_notes and not use_qk_l2norm_in_kernel
    if "stmwtensorprod" in stm_notes:
        proj_kind = "tensorprod"
    elif "stmwfactorized" in stm_notes:
        proj_kind = "factorized"
    elif "stmconconlyc4" in stm_notes:
        proj_kind = "conconlyc4"
    elif "stmconconlyc3" in stm_notes:
        proj_kind = "conconlyc3"
    elif "stmconconlyc2" in stm_notes:
        proj_kind = "conconlyc2"
    elif "stmconconlyc" in stm_notes:
        proj_kind = "conconlyc"
    elif "stmconconly" in stm_notes:
        proj_kind = "conconly"
    elif "stmconcaltc" in stm_notes:
        proj_kind = "concaltc"
    elif "stmconcalt" in stm_notes:
        proj_kind = "concalt"
    elif "stmconchaos" in stm_notes or "stmconchaossig" in stm_notes:
        proj_kind = "conchaos"
    elif "stmconcreor" in stm_notes:
        proj_kind = "concreor"
    else:
        proj_kind = "linear"
    proj_rank = str2val(stm_notes, "stmwrank", default=32, output_type=int)
    _conc_ks_flag = {
        "conconlyc4": "stmconconlyc4",
        "conconlyc3": "stmconconlyc3",
        "conconlyc2": "stmconconlyc2",
        "conconlyc": "stmconconlyc",
    }.get(proj_kind)
    conc_kernel_size = (
        str2val(stm_notes, _conc_ks_flag, default=3, output_type=int)
        if _conc_ks_flag is not None and f"{_conc_ks_flag}:" in stm_notes
        else 3
    )
    concreor = proj_kind == "concreor"
    chaos_iter_mode = None
    chaos_n_stages = None
    if proj_kind == "conchaos":
        from minipi05.stm.stm_chaos_proj import parse_stm_chaos_config

        chaos_iter_mode, chaos_n_stages = parse_stm_chaos_config(stm_notes)

    conv_notes = stm_notes.replace("pconvtype:lc2dv2", "pconvtype:cudaext2dv2br")
    if "pconvtype:" not in conv_notes:
        conv_notes += "_pconvtype:shortconv"

    return StateTensorModel(
        d_model=int(config.hidden_size),
        n_heads=_nh,
        dk=dk,
        du=du,
        dv=dv,
        gate_type=stm_gate_type,
        use_gla_gate=use_gla_gate,
        gate_logit_normalizer=gate_logit_normalizer,
        gate_low_rank_dim=gate_low_rank_dim,
        gate_clamp_min=gate_clamp_min,
        gate_cumulative=gate_cumulative,
        use_triton=use_triton,
        conv_notes=conv_notes,
        concreor=concreor,
        proj_kind=proj_kind,
        proj_rank=proj_rank,
        conc_kernel_size=conc_kernel_size,
        chaos_iter_mode=chaos_iter_mode,
        chaos_n_stages=chaos_n_stages,
        use_exp2=use_exp2,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        attn_scale=attn_scale,
        torch_chunk_size=(None if stm_torch_chunk <= 0 else stm_torch_chunk),
        backward_recompute=stm_bwd,
        gdn_conv_size=gdn_conv_size,
        train_pytorch_core_only=train_pytorch_core_only,
        use_output_gate=use_output_gate,
        fuse_output_gate=fuse_output_gate,
        use_write_gate=use_write_gate,
        fuse_write_gate=fuse_write_gate,
        use_two_betas=use_two_betas,
        use_two_betas2=use_two_betas2,
        allow_neg_eigval_write=allow_neg_eigval_write,
        use_matrix_state=use_matrix_state,
        use_delta=use_delta,
        use_delta2=use_delta2,
        use_delta3=use_delta3,
        use_fla_matrix=use_fla_matrix,
        use_fla_fused=use_fla_fused,
    )


def build_stm_layer(config: SimpleNamespace, *, layer_idx: int = 0) -> StateTensorModel:
    """Public alias used by ``minipi05.trunk.get_analyzers``."""
    return _build_stm_layer(config, layer_idx=layer_idx)