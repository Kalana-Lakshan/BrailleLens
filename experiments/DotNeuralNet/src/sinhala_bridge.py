"""Decode DotNeuralNet cell patterns to Sinhala / English via BrailleLens tables.

Reuses ``braille_cnn.labels.decode_sequence`` so indicator + vowel-sign pairs
match the rest of the project.
"""

from __future__ import annotations

import sys
from pathlib import Path

def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "braille_cnn").is_dir():
            return parent
    raise RuntimeError("Could not find BrailleLens repo root (no braille_cnn/)")


_ROOT = _repo_root()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from braille_cnn.labels import (  # noqa: E402
    CODE_TO_SINHALA,
    code_to_label,
    decode_sequence,
)

from .pattern_code import class_id_to_code, pattern_to_code  # noqa: E402


def patterns_to_codes(patterns: list[str | int]) -> list[int]:
    return [pattern_to_code(p) for p in patterns]


def decode_patterns(
    patterns: list[str | int],
    lang: str = "si",
    confidences: list[float] | None = None,
    conf_threshold: float = 0.0,
) -> str:
    """Decode an ordered list of DotNeuralNet patterns to text.

    If ``confidences`` is given, cells below ``conf_threshold`` become ``_``
    (and skip indicator pairing for that cell), matching live-pipeline behavior.
    """
    codes = patterns_to_codes(patterns)

    if confidences is None or conf_threshold <= 0:
        return decode_sequence(codes, lang=lang)

    # Build a filtered code list: low-conf → sentinel handled after decode_sequence
    # For Sinhala indicator pairs we need per-cell control, so decode manually
    # via decode_sequence on runs of good cells, inserting '_' for bad ones.
    out: list[str] = []
    i = 0
    n = len(codes)
    while i < n:
        if confidences[i] < conf_threshold:
            out.append("_")
            i += 1
            continue
        # gather a contiguous confident run
        j = i
        while j < n and confidences[j] >= conf_threshold:
            j += 1
        out.append(decode_sequence(codes[i:j], lang=lang))
        i = j
    return "".join(out)


def decode_class_ids(
    cls_ids: list[int],
    names: dict | list,
    lang: str = "si",
    confidences: list[float] | None = None,
    conf_threshold: float = 0.0,
) -> str:
    patterns = [class_id_to_code(c, names) for c in cls_ids]
    # class_id_to_code already returns codes; wrap as ints for decode_patterns
    if confidences is None or conf_threshold <= 0:
        return decode_sequence(patterns, lang=lang)
    return decode_patterns(
        patterns,
        lang=lang,
        confidences=confidences,
        conf_threshold=conf_threshold,
    )


def single_pattern_label(pattern: str | int, lang: str = "si") -> str:
    """Single-cell label (no indicator combining)."""
    return code_to_label(pattern_to_code(pattern), lang=lang)


__all__ = [
    "pattern_to_code",
    "patterns_to_codes",
    "decode_patterns",
    "decode_class_ids",
    "single_pattern_label",
    "CODE_TO_SINHALA",
]
