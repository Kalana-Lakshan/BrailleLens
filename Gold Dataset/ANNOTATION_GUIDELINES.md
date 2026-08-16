# Annotation Guidelines — LabelMe

How to hand-annotate the page photos in this folder (currently `Low quality dataset/pg-1.jpeg` … `pg-12.jpeg`) so they can feed straight into the existing Angelina-format pipeline (`experiments/DotNeuralNet/src/crop_bbox.py`, `dataset.py`). See the [project root README](../README.md) for pipeline context; this file only covers the annotation step.

---

## 1. Setup

```bash
py -3.11 -m pip install labelme
py -3.11 -m labelme "Gold Dataset/Low quality dataset"
```

This opens the folder as a browsable image list. Work through `pg-1.jpeg` → `pg-12.jpeg` in order.

---

## 2. Output convention (must match this exactly)

For every `pg-N.jpeg`, LabelMe must save `pg-N.json` **in the same folder, same basename** — this is what `crop_angelina_bbox(img_path, bbox_path)` expects (it takes the two paths as a pair). Don't rename, don't nest into a separate `labels/` folder. This mirrors how `AngelinaDataset-master/books/*/*.labeled.json` sits next to its `.jpg`.

---

## 3. What gets a box: one rectangle per physical Braille cell

- **A cell = a 2×3 dot grid, i.e. one character position** — not a word, not a line, not a multi-cell grapheme.
- Draw with LabelMe's **Rectangle** tool (`Ctrl+R` or the toolbar). Don't use polygon/circle — `crop_bbox.py` only reads `points` as a min/max box (`shape["points"].min()/.max()`), so a rectangle is required.
- **Skip blank space** — don't box gaps between words/cells where no dot is raised.
- **Skip decorative rule lines.** A few of these pages (e.g. `pg-1.jpeg`) have long unbroken rows of dots used as horizontal separators/underlines rather than text. Those aren't Braille characters — leave them unboxed.

### Box extent — the non-overlap rule from last discussion

- Box tightly around the visible raised-dot extent of **that cell only**: left edge of dot column 1 → right edge of dot column 2, top of row 1 → bottom of row 3, plus a small margin (~15–20% of one dot's radius) so the shadow/highlight halo around each dot is fully inside.
- **Stop at the midpoint gap to the neighboring cell.** Never let a box bleed into where the next cell's dots start. This is what keeps boxes non-overlapping by construction — same principle used for the DSBI dot-level boxes in `yolo_dot_detect/prepare_dataset.py` (`_estimate_box_half`), just applied at cell pitch instead of dot pitch.
- **Axis-aligned rectangles only, even if the page/line is skewed.** These photos have visible page curvature near the spine and inconsistent lighting gradients — don't try to rotate the box to follow a curved line. If a whole line is too warped/blurred to box confidently, skip that line rather than guess.
- Zoom in (scroll wheel in LabelMe) before drawing on low-contrast rows — on these photos the dots are only visible as faint shadow/highlight, not a hard edge, so working at low zoom causes boxes to drift off-center.

---

## 4. Labeling convention — what to type in the label field

Type **the single Sinhala character this one cell represents on its own**, the same way you'd naturally transcribe it — not the raw dot-number pattern, not a merged multi-cell word.

**Exception — cells with no standalone letter.** Some cells (e.g. Sinhala pillam / dependent-vowel-sign cells, which combine with the *preceding* cell to form one syllable — see `braille_cnn/labels.py`'s `CODE_TO_SINHALA` and the note in its comments) only make sense combined with the *preceding* cell and have no letter of their own. For those, don't guess a letter — type the raw dot pattern instead, wrapped the same way Angelina marks special/ambiguous symbols: **`~<dots>~`**, e.g. `~245~` for a cell with dots 2, 4, 5 raised (dot numbering: `1 4 / 2 5 / 3 6`, top-to-bottom, left-to-right, same convention as `DOT_OFFSETS` in `yolo_dot_detect/prepare_dataset.py`). This keeps every cell labeled with *something* even before Sinhala composition rules are finalized.

**Consistency matters more than correctness right now** — the same dot pattern must always get the same label string across all 12 pages. Keep a running personal cheat-sheet as you go (glyph ↔ dot pattern) instead of re-deciding each time; this becomes the seed for a `alpha_map_SI` dict (same shape as `alpha_map_RU` in `experiments/DotNeuralNet/src/utils/angelina_utils.py`), which doesn't exist yet and is required before `transform_angelina_label()` will resolve Sinhala labels.

---

## 5. Reading order

LabelMe doesn't preserve line/column order — boxes are just stored in click order. Two options:

- Annotate strictly **left-to-right, top-to-bottom** per page so click order happens to match reading order, or
- Don't worry about it and sort later — a small script bucketing boxes by y-center (line) then x-center (column) can reconstruct order post-export, the same idea as `cluster_into_cells()` in `braille_cnn/dot_detect.py`.

If in doubt, pick the first option — it costs nothing extra while annotating and avoids needing a sort script at all for a 12-page set.

---

## 6. Before moving on: sanity-check

After each page, reopen it in LabelMe (or use a quick overlay script, same idea as `yolo_dot_detect/visualize_labels.py`) and confirm:

- [ ] Every visible cell has exactly one box, no missed rows (easy to lose a row in shadowed regions)
- [ ] No two boxes overlap
- [ ] No box spans more than one cell
- [ ] Rule/separator lines are not boxed
- [ ] `pg-N.json` exists next to `pg-N.jpeg`

---

## 7. What happens after annotation (for context, not part of this step)

Once labeled, `crop_angelina_bbox(img_path, bbox_path)` in `experiments/DotNeuralNet/src/crop_bbox.py` crops each box and encodes its label via `transform_angelina_label()` into a 6-bit dot pattern, saving `pg-N_<x1>_<y1>_<x2>_<y2>_<onehot>.jpg` crops that `BrailleDataset` in `dataset.py` picks up automatically. That step is blocked until the `alpha_map_SI` dict from §4 is added — annotate now, wire up the mapping table separately.
