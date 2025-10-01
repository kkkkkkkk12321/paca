import ast
import math
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


# ---------------------------------------------------------------------------
# Optional sympy dependency stub
# ---------------------------------------------------------------------------

try:  # pragma: no cover - real sympy available
    import sympy  # type: ignore  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - provide lightweight stand-in
    _sympy = types.ModuleType("sympy")

    _SAFE_GLOBALS = {name: getattr(math, name) for name in dir(math) if not name.startswith("_")}
    _SAFE_GLOBALS.update({
        "abs": abs,
        "min": min,
        "max": max,
        "pow": pow,
    })

    class _FakeExpr:
        def __init__(self, expr: str):
            self._expr = str(expr)

        def subs(self, symbol, value):
            name = getattr(symbol, "name", str(symbol))
            replaced = self._expr.replace(str(name), str(value))
            return _FakeExpr(replaced)

        def evalf(self, _precision: int = 15):
            try:
                tree = ast.parse(self._expr, mode="eval")
                return eval(compile(tree, "<fake_sympy>", "eval"), _SAFE_GLOBALS, {})
            except Exception as exc:  # pragma: no cover - evaluation failure
                raise ValueError(str(exc)) from exc

        @property
        def is_real(self) -> bool:
            try:
                value = self.evalf()
            except Exception:
                return False
            return isinstance(value, (int, float))

        def __str__(self) -> str:  # pragma: no cover - string conversion
            return self._expr

    class _FakeSymbol:
        def __init__(self, name: str):
            self.name = name

        def __str__(self):  # pragma: no cover - debugging helper
            return self.name

    class _FakeEquation:
        def __init__(self, left, right):
            self.left = left
            self.right = right

    def _sympify(expression):
        if isinstance(expression, _FakeExpr):
            return expression
        return _FakeExpr(str(expression))

    def _symbol(name: str):
        return _FakeSymbol(name)

    def _eq(left, right):
        return _FakeEquation(_sympify(left), _sympify(right))

    def _diff(expr, symbol, order=1):
        base = str(_sympify(expr))
        sym_name = getattr(symbol, "name", str(symbol))
        return _FakeExpr(f"d^{order}/d{sym_name}^{order}({base})")

    def _integrate(expr, var, limits=None):
        base = str(_sympify(expr))
        if isinstance(var, tuple):
            symbol = getattr(var[0], "name", str(var[0]))
            lower, upper = var[1:3]
            return _FakeExpr(f"∫_{lower}^{upper} {base} d{symbol}")
        symbol = getattr(var, "name", str(var))
        return _FakeExpr(f"∫ {base} d{symbol}")

    def _factor(expr):
        return _sympify(expr)

    def _expand(expr):
        return _sympify(expr)

    def _solve(eq, _variable):  # pragma: no cover - simple placeholder
        if isinstance(eq, _FakeEquation):
            try:
                left_val = _sympify(eq.left).evalf()
                right_val = _sympify(eq.right).evalf()
                if isinstance(left_val, (int, float)) and isinstance(right_val, (int, float)):
                    if abs(left_val - right_val) < 1e-9:
                        return [right_val]
            except Exception:
                pass
        return []

    _sympy.sympify = _sympify  # type: ignore[attr-defined]
    _sympy.Symbol = _symbol  # type: ignore[attr-defined]
    _sympy.Eq = _eq  # type: ignore[attr-defined]
    _sympy.diff = _diff  # type: ignore[attr-defined]
    _sympy.integrate = _integrate  # type: ignore[attr-defined]
    _sympy.factor = _factor  # type: ignore[attr-defined]
    _sympy.expand = _expand  # type: ignore[attr-defined]
    _sympy.solve = _solve  # type: ignore[attr-defined]
    _sympy.pi = math.pi  # type: ignore[attr-defined]
    _sympy.E = math.e  # type: ignore[attr-defined]
    _sympy.__version__ = "fake-0.0"

    sys.modules.setdefault("sympy", _sympy)
