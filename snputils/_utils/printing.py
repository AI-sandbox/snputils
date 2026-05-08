from __future__ import annotations

from typing import Any, Optional, Tuple


def array_shape(value: Any) -> Optional[Tuple[int, ...]]:
    """
    Return an object's shape as a plain tuple, or None when no shape exists.
    """
    if value is None:
        return None

    shape = getattr(value, "shape", None)
    if shape is None:
        return None

    return tuple(int(dim) for dim in shape)


def format_repr(obj: Any, **fields: Any) -> str:
    """
    Build a concise repr from named metadata fields.
    """
    args = ", ".join(f"{name}={value!r}" for name, value in fields.items())
    return f"{obj.__class__.__name__}({args})"
