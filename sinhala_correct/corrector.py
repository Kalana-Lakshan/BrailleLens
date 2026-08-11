"""Lightweight Sinhala word correction: given a possibly-misread word,
suggest the most likely real word(s) it should have been.

Classic Norvig-style edit-distance spell correction (generate all 1-edit
and 2-edit variants of the input, keep the ones that are real dictionary
words, rank by frequency) -- the same general technique SinSpell/Subasa
use for Sinhala (n-gram + minimum edit distance, see README.md). Distance
is measured over raw Unicode codepoints, which doesn't perfectly respect
Sinhala's combining-vowel-sign structure (a dependent vowel sign is its
own codepoint attached to a consonant, so what looks like "one visual
character" can be 2+ codepoints) -- a known simplification, not yet a
correctness bug: it just means some edits open with a lower-level
resolution than a linguistically-aware model would use. Good enough as a
first pass; flagged here for whoever tackles a more grapheme-aware version.

This module works on decoded Sinhala TEXT (plain word strings), not on
Braille codes directly -- it's the "given a word we're not sure about,
what's the real word likely to be" layer, independent of how that
(possibly wrong) word string was produced. Wiring this against the
per-cell classifier confidence scores from braille_cnn/DotNeuralNet/etc
(so a LOW-confidence cell's alternate top-k readings get tried first,
not just single-character edits of the best guess) is the natural next
step -- not done yet, see README.md.
"""

from __future__ import annotations

from .dictionary import SinhalaDictionary


def _alphabet(dictionary: SinhalaDictionary) -> set[str]:
    """Character set to try insertions/replacements with -- derived from
    the dictionary itself rather than a hardcoded Unicode range, so it
    automatically covers exactly the codepoints (base letters AND
    combining vowel signs) that actually occur in real Sinhala words."""
    chars: set[str] = set()
    for w in dictionary.words():
        chars.update(w)
    return chars


def _edits1(word: str, alphabet: set[str]) -> set[str]:
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [a + b[1:] for a, b in splits if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
    replaces = [a + c + b[1:] for a, b in splits if b for c in alphabet]
    inserts = [a + c + b for a, b in splits for c in alphabet]
    return set(deletes + transposes + replaces + inserts)


def _edits2(word: str, alphabet: set[str]) -> set[str]:
    return {e2 for e1 in _edits1(word, alphabet) for e2 in _edits1(e1, alphabet)}


class SinhalaCorrector:
    def __init__(self, dictionary: SinhalaDictionary | None = None):
        self.dictionary = dictionary or SinhalaDictionary()
        self._alphabet = _alphabet(self.dictionary)

    def suggest(self, word: str, max_suggestions: int = 5) -> list[tuple[str, int]]:
        """Returns [(candidate_word, frequency), ...] sorted most-likely
        first. If `word` is already a known real word, returns it alone
        (nothing to correct). Empty list if no known word is within
        edit-distance 2."""
        if not word:
            return []
        if word in self.dictionary:
            return [(word, self.dictionary.frequency(word))]

        candidates = {e for e in _edits1(word, self._alphabet) if e in self.dictionary}
        if not candidates:
            candidates = {e for e in _edits2(word, self._alphabet) if e in self.dictionary}
        if not candidates:
            return []

        ranked = sorted(candidates, key=lambda w: -self.dictionary.frequency(w))
        return [(w, self.dictionary.frequency(w)) for w in ranked[:max_suggestions]]

    def best(self, word: str) -> str:
        """Single best guess -- the corrected word, or the original input
        unchanged if no correction was found (better to pass through an
        unrecognized word as-is than to silently drop/blank it)."""
        suggestions = self.suggest(word, max_suggestions=1)
        return suggestions[0][0] if suggestions else word


def _smoke() -> None:
    import time

    corrector = SinhalaCorrector()

    # a real, common word should be returned unchanged (already correct)
    assert corrector.best("මේ") == "මේ"

    # Build corrupted test cases from real dictionary words by deleting one
    # codepoint, and check whether the ORIGINAL word is recovered.
    #
    # Two checks, deliberately at different strictness levels:
    #  - top-1 (best()): the single guess actually used downstream. This
    #    will legitimately miss sometimes -- pure word-frequency ranking
    #    has no sentence context, so when a corruption is equally close
    #    (by edit distance) to two real words, it can rank a more common
    #    *unrelated* word above the less common *correct* one (confirmed
    #    empirically: "සඳහ" -> both "සහ" freq=518043 and "සඳහා" freq=294666
    #    are 1-edit away; frequency alone picks "සහ"). This is exactly the
    #    gap a context-aware model (see README.md's next-step note) would
    #    close -- not a bug in this lightweight version.
    #  - top-5 (suggest()): did the corrector at least surface the right
    #    word as a candidate, even if not ranked first. This is the more
    #    meaningful correctness bar for what this module actually promises.
    sample_words = [w for w, f in sorted(corrector.dictionary._freq.items(), key=lambda kv: -kv[1])[:500]
                     if len(w) >= 3]
    n_top1 = n_top5 = n_tested = 0
    t0 = time.time()
    for w in sample_words[:40]:
        corrupted = w[:-1]  # delete last codepoint
        if corrupted in corrector.dictionary:
            continue  # corruption happens to also be a real word -- not a correction case
        n_tested += 1
        suggestions = [cand for cand, _ in corrector.suggest(corrupted, max_suggestions=5)]
        if suggestions and suggestions[0] == w:
            n_top1 += 1
        if w in suggestions:
            n_top5 += 1
    elapsed = time.time() - t0
    print(f"top-1 recovered {n_top1}/{n_tested}, top-5 recovered {n_top5}/{n_tested} "
          f"single-deletion corruptions ({elapsed:.2f}s for {n_tested} lookups)", flush=True)
    assert n_tested > 0, "no usable test cases generated"
    assert n_top5 / n_tested >= 0.85, "even top-5 recovery rate too low -- something's off"
    print("corrector smoke OK", flush=True)


if __name__ == "__main__":
    _smoke()
