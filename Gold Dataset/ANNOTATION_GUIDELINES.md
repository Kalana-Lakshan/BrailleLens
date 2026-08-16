# Annotation Guidelines — Gold Dataset (Stage 1b)

Label **12 High quality pages** in LabelMe. The Low quality folder is the
same 12 pages under different lighting — do not label it by hand. After
High quality is done:

```bash
py -3.11 -m data_pipeline.transfer_gold_labels
```

That copies every box across with ORB+RANSAC and writes QC overlays to
`reports/gold_transfer/`. A page with too few inliers is refused and you
label that Low twin by hand.

Do this **after** DSBI + Angelina training. Gold is the final in-domain
fine-tune, not the first training set.

---

## 1. Setup

```bash
py -3.11 -m pip install labelme
py -3.11 -m labelme "Gold Dataset/High quality dataset"
```

Work through `pg-1.jpeg` → `pg-12.jpeg`.

---

## 2. Output files

For every `pg-N.jpeg`, save `pg-N.json` **in the same folder, same basename**.
No separate `labels/` folder.

---

## 3. What gets a box

- One **axis-aligned rectangle** per physical Braille **cell** (2×3 dots).
- LabelMe Rectangle tool (`Ctrl+R`). No polygons.
- Tight around that cell; stop at the midpoint gap to the neighbour.
- Skip blank space and decorative divider / ruler rows.

---

## 4. Label text = raised dot numbers

Type the raised dots, not a Sinhala letter. Examples:

| Dots raised | Label |
|---|---|
| none (blank) | `0` |
| dots 1 and 2 | `12` |
| dots 1, 3, 4, 5 | `1345` |
| all six | `123456` |

Dot numbering (same as the rest of this repo):

```
1  4
2  5
3  6
```

Why not Sinhala characters:

- The CNN's 64 classes **are** these codes (`code = sum(1 << (d-1))`).
- Sinhala → code is many-to-one; some letters span two cells.
- You do not need Sinhala Braille knowledge to see which dots are up.
- The glyph is still recovered later via `CODE_TO_SINHALA`.

---

## 5. Split (already decided — do not reshuffle files)

Both lighting variants of a page stay in the same split:

| Pages | Split |
|---|---|
| 1–8 | train |
| 9–10 | val |
| 11–12 | test |

`data_pipeline.integrate` assigns this from the page number.

---

## 6. Checklist per page

- [ ] Every visible text cell has one box
- [ ] No two boxes overlap
- [ ] No box spans two cells
- [ ] Divider rows are not boxed
- [ ] Labels are only digits `1`–`6` (or `0`)
- [ ] `pg-N.json` sits next to `pg-N.jpeg`

---

## 7. After all 12 High pages

```bash
py -3.11 -m data_pipeline.transfer_gold_labels
# inspect reports/gold_transfer/pg-*.png
py -3.11 -m data_pipeline.integrate --sources dbsi angelina gold --split-mode rebalance
py -3.11 -m data_pipeline.clean
py -3.11 -m data_pipeline.reduce
```
