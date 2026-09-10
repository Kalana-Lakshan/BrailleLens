"""CNN-* label decode and line grouping (SRS FR 9). No checkpoint required."""

from __future__ import annotations

from braille_cnn.labels import (
    INDICATOR_CODES,
    code_to_label,
    decode_sequence,
    dots_to_code,
)
from braille_cnn.recognize import _drop_ruler_lines, group_into_lines


def test_cnn01_english_space_and_letter():
    assert code_to_label(0, lang="en") == "space"
    assert code_to_label(dots_to_code((1,)), lang="en") == "a"


def test_cnn02_sinhala_blank_is_space():
    assert code_to_label(0, lang="si") == " "


def test_cnn03_unknown_code_is_hashed():
    assert code_to_label(99, lang="en") == "#99"


def test_cnn04_sinhala_combining_vowel_attaches():
    # ග (dots 2,5) + short indicator + dot1 → ගා
    ga = dots_to_code((2, 5))
    ind = next(iter(INDICATOR_CODES & {dots_to_code((3, 4, 5, 6))}))
    aa = dots_to_code((1,))
    text = decode_sequence([ga, ind, aa], lang="si")
    assert text == "ගා"


def test_cnn05_english_sequence_joins_letters():
    a = dots_to_code((1,))
    b = dots_to_code((1, 2))
    assert decode_sequence([a, b], lang="en") == "ab"


def test_cnn06_group_into_lines_reading_order():
    cells = [
        {"xyxy": (80.0, 10.0, 100.0, 40.0)},  # line 1 col 2
        {"xyxy": (10.0, 10.0, 30.0, 40.0)},  # line 1 col 1
        {"xyxy": (10.0, 80.0, 30.0, 110.0)},  # line 2
    ]
    lines = group_into_lines(cells)
    assert len(lines) == 2
    assert lines[0][0]["xyxy"][0] == 10.0
    assert lines[0][1]["xyxy"][0] == 80.0
    assert lines[1][0]["xyxy"][1] == 80.0


def test_cnn07_empty_page_has_no_lines():
    assert group_into_lines([]) == []


def test_cnn08_drop_ruler_line_keeps_mixed_text():
    text = [{"xyxy": (float(i * 20), 0.0, float(i * 20 + 10), 20.0), "code": i % 5} for i in range(16)]
    kept = _drop_ruler_lines(text, min_cells=15, top2_fraction=0.55)
    assert len(kept) == 16
    ruler = [{"xyxy": (float(i * 20), 0.0, float(i * 20 + 10), 20.0), "code": 3} for i in range(16)]
    dropped = _drop_ruler_lines(ruler, min_cells=15, top2_fraction=0.55)
    assert dropped == []
