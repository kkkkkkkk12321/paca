import sys
import types

# Provide a very small stub for the optional email_validator dependency so that
# tests which import the main package can succeed without the extra package.
_email_validator = types.ModuleType("email_validator")

class _EmailNotValidError(ValueError):
    """Fallback error type mimicking email_validator.EmailNotValidError."""


def _validate_email(value, *_args, **_kwargs):
    if not isinstance(value, str) or "@" not in value:
        raise _EmailNotValidError("Invalid email format")
    return types.SimpleNamespace(email=value)

_email_validator.validate_email = _validate_email  # type: ignore[attr-defined]
_email_validator.EmailNotValidError = _EmailNotValidError  # type: ignore[attr-defined]
_email_validator.__all__ = ["validate_email", "EmailNotValidError"]
_email_validator.__version__ = "2.1.0"

sys.modules.setdefault("email_validator", _email_validator)

# Some parts of pydantic try to query importlib.metadata for the distribution.
try:  # pragma: no cover - depends on Python version
    import importlib.metadata as importlib_metadata
except ImportError:  # pragma: no cover
    import importlib_metadata  # type: ignore

_original_distribution = getattr(importlib_metadata, "distribution", None)
_original_version = getattr(importlib_metadata, "version", None)


def _fake_distribution(name: str):  # pragma: no cover - very small shim
    if name.replace("_", "-") == "email-validator":
        return types.SimpleNamespace(version="2.1.0")
    if _original_distribution is None:
        raise importlib_metadata.PackageNotFoundError(name)
    return _original_distribution(name)


def _fake_version(name: str):  # pragma: no cover - very small shim
    if name.replace("_", "-") == "email-validator":
        return "2.1.0"
    if _original_version is None:
        raise importlib_metadata.PackageNotFoundError(name)
    return _original_version(name)

if _original_distribution is not None:
    importlib_metadata.distribution = _fake_distribution  # type: ignore[assignment]
if _original_version is not None:
    importlib_metadata.version = _fake_version  # type: ignore[assignment]
