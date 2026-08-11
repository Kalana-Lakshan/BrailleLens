# sinhala_correct — lightweight Sinhala word correction

Given a word that might have been misread (e.g. from a low-confidence
Braille cell classification), suggests the most likely real Sinhala word
it should have been — so an unclear cell doesn't have to mean an unclear
word. First step toward the "guess the word from context, even when a
character is unclear" idea discussed in chat: this piece handles "guess
the word from the dictionary"; using the classifier's own per-cell
confidence/candidates and real sentence context are later steps (see
Not yet built below).

## How it works

Classic Norvig-style edit-distance spell correction: generate every
1-edit variant of the input word (delete/transpose/replace/insert one
character), keep the ones that are real dictionary words, rank by how
common each is. If nothing is found at edit-distance 1, try edit-distance
2. This is the same general technique SinSpell/Subasa (the existing
open-source Sinhala spell checkers) use — n-gram statistics + minimum
edit distance — not a novel approach, just applied here specifically for
correcting probable Braille misreads rather than typos.

## Data source

[nlpcuom/Word-Frequency-List-for-Sinhala](https://github.com/nlpcuom/Word-Frequency-List-for-Sinhala)
(Fernando & Dias, ICON 2021) — uses the **verified** word list (280,603
words, manually checked to be real/correctly-spelled), not the larger
2.1M raw-web-corpus list, since a corrector should suggest real curated
words, not words that are themselves web-scrape misspellings.

```
@inproceedings{fernando-dias-2021-building,
  title = "Building a Linguistic Resource : A Word Frequency List for {S}inhala",
  author = "Fernando, Aloka and Dias, Gihan",
  booktitle = "Proceedings of the 18th International Conference on Natural Language Processing (ICON)",
  year = "2021",
  url = "https://aclanthology.org/2021.icon-main.74",
}
```

The source repo doesn't state an explicit license — cite the paper above
if using this beyond a research prototype, and confirm terms with the
authors before any broader/commercial use.

## Files

| File | What it does |
|---|---|
| `prepare_dictionary.py` | Converts the cloned nlpcuom repo's verified word list into this module's own `data/sinhala_words.tsv` (word + frequency, tab-separated, sorted by frequency descending). Run once. |
| `dictionary.py` | `SinhalaDictionary` — loads `data/sinhala_words.tsv` into an in-memory word→frequency lookup. |
| `corrector.py` | `SinhalaCorrector` — the actual correction logic described above. `.suggest(word, max_suggestions=5)` returns ranked candidates; `.best(word)` returns just the top one (or the original word unchanged if nothing was found, rather than silently dropping it). |

`data/sinhala_words.tsv` (~8MB) is committed directly — small enough, and
core to the feature working at all (matches this project's existing
precedent of committing model weights up to ~50MB, unlike the large raw
image datasets which stay gitignored).

## Usage

```bash
git clone https://github.com/nlpcuom/Word-Frequency-List-for-Sinhala.git /tmp/sinhala-wordfreq
py -3.11 -m sinhala_correct.prepare_dictionary --source-root /tmp/sinhala-wordfreq
```

```python
from sinhala_correct import SinhalaCorrector

c = SinhalaCorrector()
c.best("කරන්")          # -> "කරන්" (already a real word, returned unchanged)
c.suggest("සඳහ", max_suggestions=5)  # -> ranked list of real-word candidates
```

## A known, honest limitation (not a bug)

Ranking is by raw word frequency alone — no sentence context. When a
corrupted word is equally close (by edit distance) to two real words, and
the *wrong* one happens to be more common in general, frequency-only
ranking picks it. Confirmed directly: `"සඳහ"` (an intermediate corruption
of `"සඳහා"`, "for/regarding") is edit-distance-1 from *both* `"සහ"`
("and", frequency 518,043) and `"සඳහා"` (frequency 294,666) — the
corrector picks `"සහ"` first even though `"සඳහා"` was the real source
word, because "and" is simply a more common word in general, and nothing
here knows the surrounding sentence to prefer the contextually-correct
one instead.

Measured on real dictionary words with a synthetic one-character deletion
(see `corrector.py`'s own smoke test): the correct word is recovered as
the **top-1** guess ~70% of the time, but is present somewhere in the
**top-5** candidates close to 100% of the time. That gap is exactly what
a context-aware step would close.

## Not yet built

1. **Wiring into the actual per-cell classifier confidence scores.** Right
   now this module only takes a plain word *string* that's assumed to
   already be wrong — it doesn't yet know which specific cell(s) in that
   word were low-confidence, or what the classifier's *other* candidate
   readings for that cell were (top-k, not just the single best guess).
   Using that would let correction try the classifier's own real
   alternate readings at the uncertain position(s) first, not just
   generic single-character edits — a much stronger signal than "any
   1-edit variant of the final guess."
2. **Sentence/context awareness**, to fix the top-1-ranking limitation
   above — e.g. a small local Sinhala language model
   ([keshan/sinhala-gpt2](https://huggingface.co/keshan/sinhala-gpt2) was
   the specific candidate discussed) that can score "how likely is this
   word given the words already read before it," not just "how common is
   this word in isolation." Still meant to run locally/fast for the live
   reading loop, not as a cloud LLM call.
3. **Sinhala-aware edit distance.** Distance is currently measured over
   raw Unicode codepoints, which doesn't respect that a dependent vowel
   sign is its own codepoint attached to a base consonant — so a
   "one-codepoint edit" doesn't always correspond to what a reader would
   consider "one character changed." Not yet a demonstrated problem, just
   an acknowledged simplification worth revisiting if correction quality
   plateaus.
