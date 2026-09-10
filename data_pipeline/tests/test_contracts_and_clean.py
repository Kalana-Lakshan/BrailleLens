"""DI-* data-integrity cases for data_pipeline (SRS FR 10). No UI, no DBMS."""

from __future__ import annotations

import pandas as pd
import pytest

from data_pipeline.clean import clean, leakage_report
from data_pipeline.contracts import (
    MANIFEST_COLUMNS,
    CellRow,
    code_to_dot_string,
    code_to_dots,
    dot_string_to_code,
    dots_to_code,
    validate_manifest,
    write_manifest,
    read_manifest,
)


def _row(**overrides) -> dict:
    base = dict(
        source="gold",
        image_path="Gold Dataset/pg-1.jpeg",
        page_group="gold:pg-1",
        book="gold",
        page="1",
        side="",
        split="train",
        x0=10.0,
        y0=10.0,
        x1=30.0,
        y1=40.0,
        code=19,
        dots="125",
        img_w=800,
        img_h=600,
        dot_pitch_px=15.0,
    )
    base.update(overrides)
    return base


def test_di01_cellrow_matches_manifest_columns():
    assert [f for f in CellRow.__dataclass_fields__] == MANIFEST_COLUMNS


def test_di02_dot_string_roundtrip_and_blank():
    assert dot_string_to_code("1345") == dots_to_code((1, 0, 1, 1, 1, 0))
    assert code_to_dot_string(0) == "0"
    assert dot_string_to_code("") == 0
    assert code_to_dots(63) == (1, 1, 1, 1, 1, 1)


def test_di03_bad_dot_string_raises():
    with pytest.raises(ValueError):
        dot_string_to_code("17")
    with pytest.raises(ValueError):
        dot_string_to_code("11")


def test_di04_validate_rejects_code_outside_0_63():
    frame = pd.DataFrame([_row(code=99, dots="125")])
    problems = validate_manifest(frame)
    assert any("0-63" in p for p in problems)


def test_di05_validate_detects_page_group_leakage():
    frame = pd.DataFrame(
        [
            _row(split="train"),
            _row(split="test", x0=40.0, x1=60.0),
        ]
    )
    problems = validate_manifest(frame)
    assert any("page_group" in p for p in problems)
    assert leakage_report(frame) == ["gold:pg-1"]


def test_di06_clean_drops_inverted_out_of_page_and_duplicates():
    rows = [
        _row(),  # keep
        _row(x0=50.0, x1=40.0, y0=10.0, y1=40.0),  # inverted
        _row(x0=900.0, y0=10.0, x1=950.0, y1=40.0),  # far outside page
        _row(),  # duplicate of first
        _row(dots="12", code=19, x0=70.0, x1=90.0),  # dots/code disagree
    ]
    frame = pd.DataFrame(rows)
    cleaned, log = clean(frame, drop_rulers=False, check_images=False)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["code"] == 19
    removed = {rule: n for rule, n, _ in log.entries}
    assert removed["Zero-area or inverted boxes"] == 1
    assert removed["Duplicate cell annotations"] == 1
    assert removed["dots / code disagreement"] == 1
    assert removed["Boxes outside the page"] == 1


def test_di07_clean_then_validate_no_leak_on_disjoint_groups():
    frame = pd.DataFrame(
        [
            _row(page_group="gold:pg-1", split="train"),
            _row(
                page_group="gold:pg-2",
                split="test",
                image_path="Gold Dataset/pg-2.jpeg",
                x0=12.0,
                x1=32.0,
            ),
        ]
    )
    cleaned, _ = clean(frame, drop_rulers=False, check_images=False)
    assert leakage_report(cleaned) == []
    assert validate_manifest(cleaned) == []


def test_di08_manifest_roundtrip_preserves_columns(tmp_path):
    path = tmp_path / "manifest.csv"
    write_manifest([_row()], path)
    frame = read_manifest(path)
    assert list(frame.columns) == MANIFEST_COLUMNS
    assert int(frame.iloc[0]["code"]) == 19
