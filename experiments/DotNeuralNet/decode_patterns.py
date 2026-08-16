"""Decode DotNeuralNet 6-bit patterns to Sinhala / English (Part 1).

From BrailleLens repo root:

    py -3.11 experiments/DotNeuralNet/decode_patterns.py --patterns 100000 101011 110110
    py -3.11 experiments/DotNeuralNet/decode_patterns.py --patterns 100000 101000 --lang en
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "braille_cnn").is_dir():
            return parent
    raise RuntimeError("Could not find BrailleLens repo root (no braille_cnn/)")


_ROOT = _repo_root()
_DNN = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_DNN) not in sys.path:
    sys.path.insert(0, str(_DNN))


def main():
    # Windows consoles are often cp1252 — force UTF-8 so Sinhala prints.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    p = argparse.ArgumentParser(
        description="Decode DotNeuralNet cell patterns using BrailleLens Sinhala tables"
    )
    p.add_argument(
        "--patterns",
        nargs="+",
        required=True,
        help="6-bit patterns in d1..d6 order, e.g. 100000 101011",
    )
    p.add_argument("--lang", choices=("si", "en"), default="si")
    args = p.parse_args()

    from src.pattern_code import code_to_pattern, pattern_to_code
    from src.sinhala_bridge import decode_patterns, single_pattern_label

    print(f"lang={args.lang}")
    print("-" * 40)
    for pat in args.patterns:
        code = pattern_to_code(pat)
        print(
            f"  {pat} -> code={code:2d} ({code_to_pattern(code)})  "
            f"single={single_pattern_label(pat, args.lang)!r}"
        )
    print("-" * 40)
    print(f"sequence: {decode_patterns(args.patterns, lang=args.lang)}")


if __name__ == "__main__":
    main()
