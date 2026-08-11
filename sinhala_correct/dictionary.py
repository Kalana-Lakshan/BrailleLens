"""Loads the prepared Sinhala word-frequency dictionary (see
prepare_dictionary.py) into a fast in-memory lookup: is this a real word,
and how common is it relative to other real words.
"""

from __future__ import annotations

from pathlib import Path

_HERE = Path(__file__).resolve().parent
_DEFAULT_PATH = _HERE / "data" / "sinhala_words.tsv"


class SinhalaDictionary:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else _DEFAULT_PATH
        if not self.path.exists():
            raise FileNotFoundError(
                f"Dictionary not found: {self.path}\n"
                "Run first: py -3.11 -m sinhala_correct.prepare_dictionary --source-root <path>"
            )
        self._freq: dict[str, int] = {}
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                word, _, count = line.rstrip("\n").partition("\t")
                if word and count.isdigit():
                    self._freq[word] = int(count)

    def __len__(self) -> int:
        return len(self._freq)

    def __contains__(self, word: str) -> bool:
        return word in self._freq

    def frequency(self, word: str) -> int:
        """0 if not a known word."""
        return self._freq.get(word, 0)

    def words(self):
        return self._freq.keys()


def _smoke() -> None:
    d = SinhalaDictionary()
    print(f"loaded {len(d)} words", flush=True)
    assert len(d) > 100_000, "dictionary looks too small -- did prepare_dictionary run correctly?"
    # sanity: the two most common Sinhala function words should be present
    # and clearly outrank a rare/unlikely string.
    assert "මේ" in d
    assert d.frequency("මේ") > d.frequency("xyz_not_a_word")
    print("dictionary smoke OK", flush=True)


if __name__ == "__main__":
    _smoke()
