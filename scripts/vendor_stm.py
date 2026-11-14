#!/usr/bin/env python
"""Copy STM / conv layers from thepebbletrail_official into minipi05.stm (one-time vendoring)."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

SRC = Path(r"C:\Users\Luca\work\thepebbletrail_official\neuron_utils\_more_layers")
DST = Path(r"C:\Users\Luca\work\minipi05\minipi05\stm")

FILES = [
    "state_tensor_model.py",
    "stm_delta_ut.py",
    "stm_triton.py",
    "stm_chaos_proj.py",
    "some_convs.py",
    "some_convs_cuda.py",
    "circ_channel_conv.py",
]

REPLACEMENTS = [
    ("thepebbletrail_official.neuron_utils._more_layers", "minipi05.stm"),
    ("thepebbletrail_official.neuron_utils.gateddeltanet", "minipi05.gateddeltanet"),
    (
        "from pyaromatics.stay_organized.utils import str2val",
        "from minipi05.stm.str2val import str2val",
    ),
    (
        "from pyaromatics.stay_organized.utils import str2val",
        "from minipi05.stm.str2val import str2val",
    ),
]

SOME_CONVS_STUB_HEADER = '''"""Vendored from PebbleTrail (inference-focused stubs for optional CUDA ext)."""
from typing import Any, Optional
import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers.activations import ACT2CLS

from minipi05.stm.str2val import str2val

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    TRITON_AVAILABLE = False

possible_output_collapse_methods = [
    'convs', 'meanpool', 'stdpool', 'maxpool', 'none', 'meanstdpool', 'avgproportionpool'
]

def light_causal_conv2d_fn(*args, **kwargs):
    raise NotImplementedError("lc2dv2 CUDA path not vendored for minipi05 inference")

def _can_use_cuda_kernel(*args, **kwargs):
    return False

def _two_layer_lc2dv2_gudermanlu_fwd(*args, **kwargs):
    raise NotImplementedError

def lc2dv2_triton_blockred_fn(*args, **kwargs):
    raise NotImplementedError

def lc2dv2_patch_mm_triton_fn(*args, **kwargs):
    raise NotImplementedError

def lc2dv2_patch_mm_triton_bf16_fn(*args, **kwargs):
    raise NotImplementedError

def lc2dv2_patch_mm_triton_chunked_fn(*args, **kwargs):
    raise NotImplementedError

_symm_conv1d_ext = None
SYMM_CONV1D_EXT_AVAILABLE = False
_symm_conv1d_ext_available = lambda: False

def _lc2dv2_ext(*args, **kwargs):
    raise NotImplementedError

_lc2dv2_ext_vec4 = _lc2dv2_ext
_lc2dv2_ext_vec4_2stage_warp = _lc2dv2_ext
_lc2dv2_ext_blockred = _lc2dv2_ext
_lc2dv2_ext_2layer_fwd = _lc2dv2_ext
_lc2dv2_ext_2stage = _lc2dv2_ext
_lc2dv2_ext_2stage_warp = _lc2dv2_ext
_lc2dv2_ext_2stage_warp_blockred = _lc2dv2_ext
_lc2dv2_ext_prod512 = _lc2dv2_ext
_lc2dv2_ext_amp = _lc2dv2_ext
_lc2dv2_ext_amp_blockred = _lc2dv2_ext
_lc2dv2_amp_blockred_available = lambda: False
_lc2dv2_ext_available = lambda: False

LC2DV2_EXT_AVAILABLE = False

_HALF = (torch.float16, torch.bfloat16)

try:
    from causal_conv1d import causal_conv1d_fn as _causal_conv1d_fn
except ImportError:
    _causal_conv1d_fn = None

'''


def _patch_some_convs(text: str) -> str:
    """Replace heavy CUDA import block with stub header tail join."""
    marker_start = 'from thepebbletrail_official.neuron_utils._more_layers.some_convs_cuda import ('
    if marker_start not in text and 'from minipi05.stm.some_convs_cuda import' not in text:
        if 'possible_output_collapse_methods' in text[:3000]:
            # Already patched or uses new layout
            idx = text.find("def concalt_perm")
            if idx > 0:
                return SOME_CONVS_STUB_HEADER + text[idx:]
        return text
    # strip from first cuda import through causal_conv1d try block
    start = text.find('from ')
    for needle in (
        'from thepebbletrail_official.neuron_utils._more_layers.some_convs_cuda import',
        'from minipi05.stm.some_convs_cuda import',
    ):
        if needle in text:
            start = text.find(needle)
            break
    idx = text.find("def concalt_perm")
    if idx < 0:
        return text
    return SOME_CONVS_STUB_HEADER + text[idx:]


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        src = SRC / name
        if not src.is_file():
            raise SystemExit(f"Missing source file: {src}")
        text = src.read_text(encoding="utf-8")
        for old, new in REPLACEMENTS:
            text = text.replace(old, new)
        if name == "some_convs.py":
            text = _patch_some_convs(text)
            # STM default path uses shortconv only; silence noisy debug print.
            text = text.replace(
                "    print('kernel kw:', kernel_size_w, 'kt:', kernel_size_t, 'conv_type:', conv_type)\n",
                "",
            )
        (DST / name).write_text(text, encoding="utf-8")
        print(f"wrote {DST / name}")

    # Stubs for optional CUDA training kernels (not needed for VLA inference).
    stub = DST / "stm_chunked_bilinear_ext.py"
    stub.write_text(
        '"""Optional CUDA bilinear ext (stub — not built in minipi05)."""\n'
        "def extension_available():\n    return False\n",
        encoding="utf-8",
    )
    print(f"wrote {stub}")


if __name__ == "__main__":
    main()