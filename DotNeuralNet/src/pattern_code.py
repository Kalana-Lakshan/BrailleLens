"""Map DotNeuralNet YOLO cell class names to BrailleLens integer codes.

DotNeuralNet names cells with a 6-character bit string ``d1 d2 d3 d4 d5 d6``
(e.g. ``"100110"``). BrailleLens uses the same DSBI order:

    code = sum(1 << (i) for i, bit in enumerate(d1..d6) if bit == 1)

Important: YOLO *class index* is ``int(name, 2)`` in their yaml, which is
**not** the BrailleLens code (bit significance differs). Always convert via
the pattern string from ``model.names[cls_id]``.
"""

from __future__ import annotations


def pattern_to_code(pattern: str | int) -> int:
    """Convert a 6-bit pattern string (or 0-63 if already a BrailleLens code) to int.

    Accepts:
      - ``"100110"`` / ``"100110 "`` — DotNeuralNet class name
      - ``0``-``63`` int — treated as an already-decoded BrailleLens code
    """
    if isinstance(pattern, int):
        if 0 <= pattern <= 63:
            return pattern
        raise ValueError(f"code out of range: {pattern}")

    s = str(pattern).strip()
    if len(s) != 6 or any(c not in "01" for c in s):
        raise ValueError(f"expected 6-bit pattern like '100110', got {pattern!r}")

    code = 0
    for i, bit in enumerate(s):
        if bit == "1":
            code |= 1 << i
    return code


def code_to_pattern(code: int) -> str:
    """Inverse of pattern_to_code — useful for debugging."""
    if not 0 <= code <= 63:
        raise ValueError(f"code out of range: {code}")
    return "".join("1" if (code >> i) & 1 else "0" for i in range(6))


def class_id_to_code(cls_id: int, names: dict | list | None) -> int:
    """Resolve a YOLO class id to a BrailleLens code using model.names."""
    if names is None:
        raise ValueError("model.names is required to map class ids to patterns")

    if isinstance(names, dict):
        name = names.get(int(cls_id), names.get(str(cls_id)))
    else:
        name = names[int(cls_id)]

    if name is None:
        raise KeyError(f"class id {cls_id} not in model.names")
    return pattern_to_code(str(name))
