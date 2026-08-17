"""Load and run distilled edge VLA students (minipi05)."""

from minipi05.load import load_policy, load_controller, ensure_checkpoint

__all__ = ["load_policy", "load_controller", "ensure_checkpoint"]

__version__ = "0.1.0"
