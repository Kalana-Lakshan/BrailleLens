"""CLI: map DotNeuralNet pattern strings → Sinhala (Part 1).

Examples (from BrailleLens repo root):

    py -3.11 experiments/DotNeuralNet/src/sinhala_cli.py --patterns 100000 101010 110110
    py -3.11 experiments/DotNeuralNet/src/sinhala_cli.py --patterns 011000 100000 --lang en
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
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Allow both `python -m DotNeuralNet.src.sinhala_cli` and script-style runs
_DNN_SRC = Path(__file__).resolve().parent
if str(_DNN_SRC.parent) not in sys.path:
    sys.path.insert(0, str(_DNN_SRC.parent))


def main():
    p = argparse.ArgumentParser(
        description="Decode DotNeuralNet 6-bit cell patterns to Sinhala/English"
    )
    p.add_argument(
        "--patterns",
        nargs="+",
        required=True,
        help="Space-separated patterns, e.g. 100000 101011",
    )
    p.add_argument("--lang", choices=("si", "en"), default="si")
    args = p.parse_args()

    from src.pattern_code import pattern_to_code, code_to_pattern
    from src.sinhala_bridge import decode_patterns, single_pattern_label

    print(f"lang={args.lang}")
    print("-" * 40)
    codes = []
    for pat in args.patterns:
        code = pattern_to_code(pat)
        codes.append(code)
        print(f"  {pat} -> code {code:2d} ({code_to_pattern(code)})  single={single_pattern_label(pat, args.lang)!r}")
    print("-" * 40)
    text = decode_patterns(args.patterns, lang=args.lang)
    print(f"sequence: {text}")


if __name__ == "__main__":
    main()
