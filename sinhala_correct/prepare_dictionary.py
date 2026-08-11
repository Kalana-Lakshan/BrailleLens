"""Step 1 -- Fetch and normalize the Sinhala word-frequency dictionary this
module's correction logic runs against.

Source: nlpcuom/Word-Frequency-List-for-Sinhala (Fernando & Dias, ICON 2021
-- see README.md's Citation). Uses the *verified* word list (280,603 words,
manually checked to be real/correctly-spelled Sinhala), not the larger 2.1M
raw-web-corpus list -- for a spell-*corrector*, suggesting a correction
toward a real curated word matters more than covering every word ever
scraped from the web, some of which are themselves misspellings.

Usage (from repo root):
    git clone https://github.com/nlpcuom/Word-Frequency-List-for-Sinhala.git /tmp/sinhala-wordfreq
    py -3.11 -m sinhala_correct.prepare_dictionary --source-root /tmp/sinhala-wordfreq
"""

from __future__ import annotations

import argparse
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_OUT = _HERE / "data" / "sinhala_words.tsv"
_SOURCE_FILENAME = "verified_word_list_200K.si"


def convert(source_root: Path, out_path: Path) -> int:
    src = source_root / _SOURCE_FILENAME
    if not src.exists():
        raise SystemExit(
            f"{_SOURCE_FILENAME} not found under {source_root}\n"
            "Clone first: git clone https://github.com/nlpcuom/Word-Frequency-List-for-Sinhala.git"
        )

    # word + freq. Dedupe (keep max freq seen) since a source line glitch
    # or re-run shouldn't silently double-count a word.
    freq: dict[str, int] = {}
    with open(src, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit(" ", 1)
            if len(parts) != 2:
                continue
            word, count_str = parts
            word = word.strip()
            if not word or not count_str.isdigit():
                continue
            freq[word] = max(freq.get(word, 0), int(count_str))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for word, count in sorted(freq.items(), key=lambda kv: -kv[1]):
            f.write(f"{word}\t{count}\n")

    return len(freq)


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare the Sinhala word-frequency dictionary")
    p.add_argument("--source-root", type=Path, required=True,
                    help="Path to cloned nlpcuom/Word-Frequency-List-for-Sinhala")
    p.add_argument("--out", type=Path, default=_OUT)
    args = p.parse_args()

    n = convert(args.source_root, args.out)
    print(f"Wrote {n} words -> {args.out}")


if __name__ == "__main__":
    main()
