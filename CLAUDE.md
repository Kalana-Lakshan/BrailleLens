# Interactive Braille Learning Application — Project Context

CS3501 Data Science and Engineering Project, University of Moratuwa. Group 15, Project ID P06.
Mentor: Dr. Sandareka Wickramanayake. See `Project Proposal - Group 15.pdf` for the full proposal.

## What this project is

An AI-powered Braille learning/self-assessment system built on NUS AiSee smart glasses. The glasses'
camera watches a learner's finger on a Braille page; a CV model identifies the character being
touched. Two modes:
- **Learning Mode** — reads the recognized character aloud (TTS).
- **Testing Mode** — listens to the learner's spoken answer (ASR) and gives instant feedback.

Goal: independent, hands-free Braille practice without needing an instructor present.

**Team:** Kuruppu H.A. (data/model side — this repo's CV work so far), Lakshan H.M.K. (CV
detection/glasses integration), Lawanya K.K.H.G. (app workflow, ASR/TTS integration, evaluation).

## Reference literature (`docs/`)

Three papers studied before starting implementation:
- **DSBI paper** — built the first public Braille image dataset (114 scanned double-sided pages);
  classical dot-detection (segmentation vs Haar+Adaboost), not deep learning.
- **CNN paper** (Ovodov, "Optical Braille Recognition Using Object Detection CNN") — the strongest
  architectural precedent for this project: a single-stage RetinaNet-style detector that finds and
  classifies whole Braille cells directly in one pass on real phone photos (perspective distortion,
  page curvature). Introduced the Angelina dataset for that purpose.
- **Sinhala Braille paper** (Perera & Wanniarachchi) — classical (non-DL) MATLAB pipeline for
  Sinhala/English Grade-1 Braille translation; gave the dot-pattern → character lookup table this
  project's Sinhala labels were transcribed from (Figure 12, by hand — see caveat below).

## What's built so far: `braille_cnn/`

A from-scratch PyTorch pipeline for the **first sub-problem**: classify a single Braille cell image
into one of 64 dot-pattern classes (2×3 grid, 6 bits). Letter/language decoding is kept as a
*separate* lookup step from classification, not baked into the model.

| File | What it does |
|---|---|
| `render.py` | Procedurally generates synthetic Braille cell images (Gaussian-bump dots + directional shading + rotation + **perspective/homography warp** + blur/noise). `max_perspective` controls warp strength (0 = off). |
| `labels.py` | `CODE_TO_LETTER` (Grade-1 English, all 64), `CODE_TO_SINHALA` (24 of 64 — see caveat below). `code_to_label(code, lang="en"\|"si")`. |
| `dataset.py` | `SyntheticBrailleDataset` — infinite-variety train split (fresh random render per call), seeded/reproducible test split. |
| `cnn.py` | `SimpleBrailleCNN` — small 3-conv-block CNN, 64-class output. |
| `train.py` | Trains on synthetic data only. Checkpoint: `checkpoints/braille_cnn_best.pt`. |
| `dbsi_dataset.py` | `DBSIDataset` — loads real cells from `data DBSI/` using its ground-truth per-dot pixel annotations. **Eagerly decodes/crops every cell once at construction** (uint8 in-memory tensor) — do NOT change this back to lazy per-`__getitem__` loading, see Known Issues. |
| `finetune_dbsi.py` | Fine-tunes the synthetic checkpoint on DBSI's real training split. Checkpoint: `checkpoints/braille_cnn_dbsi_finetuned.pt`. |
| `eval_dbsi.py` | Evaluates a checkpoint on DBSI (`--sides recto\|verso\|recto,verso`). |
| `eval_perspective.py` | Evaluates checkpoints on synthetic data with/without perspective warp. |
| `inspect_errors.py` | Dumps real crop images for any (true-class, predicted-class) pair — diagnostic tool for investigating confusions. |
| `show_confusion.py` | Turns a saved `.npy` confusion matrix into a readable CSV + heatmap PNG (`.npy` isn't human-readable directly). |
| `RESULTS.md` | **Full experiment log — read this for details on every finding below.** |

## Key findings so far (full detail in `braille_cnn/RESULTS.md`)

1. Synthetic-only training reaches 100% on synthetic test data — validates the pipeline works, but
   isn't informative about real-world readiness on its own.
2. **Zero-shot on real data (DBSI) collapses to 51.2%** — confirms a real synthetic→real domain gap.
3. **Fine-tuning on DBSI's real train split (26 pages) fixes it: 98.44%** on the full DBSI test set
   (71,250 real cells) — validates the proposal's "public dataset → fine-tune" pipeline works well.
4. Recto vs verso: verso is *slightly better* than recto (98.65% vs 98.24%) on DBSI — contradicts the
   papers' framing of verso as harder, at least for this scanner dataset.
5. The one lingering DBSI confusion (`#18` dots-2,5 → `j`/`h`) was traced to a probable **borderline
   ground-truth label** in DBSI itself (a faint real embossing feature scored as "no dot" by the
   human annotator), not a bug in the cropping/training code — confirmed by testing whether tighter
   crop margins removed it (they didn't).
6. **The fine-tuned checkpoint has catastrophically forgotten the synthetic domain** (14% accuracy on
   synthetic images, unrelated to perspective) — fine-tuning on real data only, with no synthetic
   data mixed back in, overwrote its ability to handle synthetic-style images. Don't assume this
   checkpoint generalizes to any new visual domain (including Angelina, later) without addressing this.
7. **Perspective/skew was never tested until added deliberately**: DBSI's images are already
   de-skewed by the dataset itself (raw skew was only ~0.25° average anyway); the synthetic renderer
   only had mild ±10° in-plane rotation. Added a real homography-based perspective warp to
   `render.py` — the synthetic-only checkpoint drops **99.97% → 87.47%** under it. Not yet retrained
   to fix this.

## Known issues / gotchas

- **`DBSIDataset` must eagerly decode/cache crops at construction, not lazily per-sample.** An
  earlier version cached only the last-opened page image; with `shuffle=True` during training this
  caused a full-resolution JPEG re-decode on almost every sample (87+ minutes and not finishing —
  looked like a hardware problem, wasn't). Fixed version builds all crops once upfront (~45s total
  for train+test). If DBSI loading ever seems slow again, check this first before assuming hardware.
- The `data DBSI/` folder is ~370MB (573 files) — large enough that whether to commit it to a shared
  git repo (vs. `.gitignore` it and document how to re-download DSBI) is a real tradeoff, not decided
  in favor of either yet as of this writing.
- `CODE_TO_SINHALA` in `labels.py` is a **hand transcription** from a compressed chart image (Figure
  12 in the Sinhala paper) — only 24 of 64 codes are filled in (the unambiguous ones); the rest were
  deliberately left out (punctuation, Grade-2 English word contractions, and combining vowel-sign
  codes that render as broken placeholder glyphs even in the source PDF itself). Worth a native
  Sinhala speaker's proofread before relying on it for real TTS output.
- Windows + PowerShell/Git-Bash environment; PDF page-rendering (`Read` tool on a page range) needs
  `pdftoppm` (poppler) on `PATH` — if missing, extract text via `pdftotext` instead, or add poppler's
  `Library/bin` to `PATH` for that Bash call.

## Open design question (not yet built)

**Finger occlusion**: during actual reading, the student's finger covers the exact dots the camera
needs to see — none of the recognition work above accounts for this (the papers it's based on are
all for sighted users scanning a clean, finger-free page). Current thinking (not yet implemented):
pre-scan the whole page once before a lesson (finger clear) to build a full `(row, col) → character`
lookup table using page-level detection, then during the lesson only track the fingertip
(MediaPipe) and detect the local grid alignment per live frame (works even under partial occlusion,
since grid lines are visible in nearby uncovered cells) to know which `(row, col)` is currently being
touched — look up its identity rather than re-recognizing it live.

## Deliberately deferred

The **Angelina dataset** (real handheld phone photos, perspective distortion, page curvature —
described in the CNN paper) is intentionally being saved for a later, bigger evaluation round,
rather than pulled in now. It only has full-page bounding-box annotations (not pre-cropped cells like
DBSI), so using it will likely mean either a cropping step or moving toward an object-detection
architecture (RetinaNet-style, per the Ovodov paper) instead of the current per-cell classifier.
