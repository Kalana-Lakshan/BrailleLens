"""Sanity-check script for the Sinhala Braille label tables in labels.py.

Run with:
    python -m braille_cnn.check_labels

On Windows, set the terminal to UTF-8 first:
    chcp 65001

Prints:
  1. All 64 codes (0-63) with their Sinhala label or #code if unmapped.
  2. Any duplicate Sinhala labels (two different codes mapping to the same letter).
  3. The two indicator codes and all two-cell pair mappings.
  4. A quick decode_sequence() round-trip test.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from .labels import (
    CODE_TO_SINHALA,
    TWO_CELL_VOWEL_SIGNS,
    INDICATOR_CODES,
    decode_sequence,
    dots_to_code,
)


def main():
    print("=" * 60)
    print("CODE_TO_SINHALA — all 64 codes")
    print("=" * 60)
    for code in range(64):
        label = CODE_TO_SINHALA.get(code, f"#{code}")
        dots = [d for d in range(1, 7) if code & (1 << (d - 1))]
        indicator = " ◄ INDICATOR" if code in INDICATOR_CODES else ""
        print(f"  code {code:2d}  dots{str(dots):<20}  ->  {label}{indicator}")

    print()
    print("=" * 60)
    print("Duplicate Sinhala labels (if any)")
    print("=" * 60)
    seen: dict[str, list[int]] = {}
    for code, label in CODE_TO_SINHALA.items():
        seen.setdefault(label, []).append(code)
    found_dup = False
    for label, codes in seen.items():
        if len(codes) > 1:
            print(f"  DUPLICATE  '{label}'  codes={codes}")
            found_dup = True
    if not found_dup:
        print("  (none — good)")

    print()
    print("=" * 60)
    print("Indicator codes")
    print("=" * 60)
    for ic in sorted(INDICATOR_CODES):
        dots = [d for d in range(1, 7) if ic & (1 << (d - 1))]
        print(f"  indicator code {ic}  dots{dots}")

    print()
    print("=" * 60)
    print("Two-cell vowel sign table")
    print("=" * 60)
    for (ind, mod), sign in sorted(TWO_CELL_VOWEL_SIGNS.items()):
        ind_dots = [d for d in range(1, 7) if ind & (1 << (d - 1))]
        mod_dots = [d for d in range(1, 7) if mod & (1 << (d - 1))]
        print(f"  ({ind},{mod})  ind_dots={ind_dots}  mod_dots={mod_dots}  →  '{sign}'")

    print()
    print("=" * 60)
    print("decode_sequence() round-trip tests")
    print("=" * 60)

    # Test 1: ක ම ල  (ක=19, ම=28, ල=37)
    seq1 = [
        dots_to_code((1, 2, 5)),    # ක  code 19
        dots_to_code((3, 4, 5)),    # ම  code 28
        dots_to_code((1, 3, 6)),    # ල  code 37
    ]
    result1 = decode_sequence(seq1, lang="si")
    status1 = "OK" if result1 == "කමල" else f"MISMATCH (got {result1!r})"
    print(f"  codes={seq1}  ->  '{result1}'  [{status1}]")

    # Test 2: ක + ා  (indicator=60, modifier=1)
    _IND_A = 60                         # dots(3,4,5,6)
    _MOD_AA = dots_to_code((1,))        # ා modifier
    seq2 = [dots_to_code((1, 2, 5)), _IND_A, _MOD_AA]
    result2 = decode_sequence(seq2, lang="si")
    expected2 = "කා"
    status2 = "OK" if result2 == expected2 else f"MISMATCH (got {result2!r})"
    print(f"  codes={seq2}  ->  '{result2}'  [{status2}]")

    # Test 3: බ [space] ම  (බ=1, space=0, ම=28)
    seq3 = [dots_to_code((1,)), 0, dots_to_code((3, 4, 5))]
    result3 = decode_sequence(seq3, lang="si")
    expected3 = "බ ම"
    status3 = "OK" if result3 == expected3 else f"MISMATCH (got {result3!r})"
    print(f"  codes={seq3}  ->  '{result3}'  [{status3}]")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
