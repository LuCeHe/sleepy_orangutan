"""Parse PebbleTrail-style ``key:value`` tokens from ``config.notes``."""

from __future__ import annotations

import re
from typing import Any, TypeVar

T = TypeVar("T")


def str2val(notes: str, key: str, default: Any = "", output_type: type = str) -> Any:
    m = re.search(rf"{re.escape(key)}:([^_:]+)", notes or "")
    if not m:
        return output_type(default) if output_type is not str else default
    raw = m.group(1)
    if output_type is bool:
        return raw.lower() in ("1", "true", "yes", "on")
    if output_type is int:
        return int(float(raw))
    if output_type is float:
        return float(raw)
    return output_type(raw)