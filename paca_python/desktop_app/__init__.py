"""PACA v5 Desktop Application package."""

__version__ = "5.0.0"
__author__ = "PACA Development Team"

__all__ = ["PacaDesktopApp", "main"]


def __getattr__(name):
    if name in {"PacaDesktopApp", "main"}:
        from . import main as _main  # Lazy import to avoid optional dependency issues

        return getattr(_main, name)
    raise AttributeError(f"module 'desktop_app' has no attribute '{name}'")
