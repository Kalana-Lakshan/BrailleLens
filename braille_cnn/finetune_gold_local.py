"""Stage 4c-gold, local-machine variant -- fine-tune the character CNN on
real Gold cell crops without the full data_pipeline artifacts.

braille_cnn/finetune_gold.py already exists and is the *intended* Stage 4c
script (reads data_pipeline/crops/crops_{train,val}.npz, sources=["gold"] vs
["dbsi","angelina"], via CropDataset) -- do not confuse this file with that
one. That script's own _has_gold() gate prints the full intended pipeline
(transfer_gold_labels -> integrate --sources dbsi angelina gold -> clean ->
reduce -> finetune_gold) and refuses to run without it. This machine can't
run that pipeline: the raw Angelina source folder ("data Angelina/books")
doesn't exist locally at all, and data_pipeline/crops/*.npz (the Stage 2c
archives braille_cnn_mixed.pt was actually trained on, presumably built on
Colab) isn't present either -- confirmed by inspection, not assumed. This
script is a stand-in for use on a machine in that state: it crops gold cells
directly from the 12 LabelMe JSONs and mixes them with a freshly-built,
in-memory DBSI dataset (raw DBSI *is* present locally) instead of the
missing crops_train.npz. Once data_pipeline/crops/ exists here (or this runs
on a machine that has it, e.g. Colab), prefer the real
braille_cnn/finetune_gold.py instead -- it also gets Angelina into the mix
and into the regression check, which this script cannot do at all.

This fine-tunes the *classifier* (braille_cnn_mixed.pt); cell_detect/finetune_gold.py
is the analogous script for the cell *detector*. manifest_clean.csv currently
has zero gold rows (data_pipeline/manifests/manifest_clean.csv only has
dbsi/angelina -- confirmed by inspection), so braille_cnn_mixed.pt has never
seen a single real gold crop. That is the actual domain gap behind
eval_gold_text.py's low letters-only accuracy, separate from the
space/detection-recall gap that recognize.py's word-gap heuristic addresses.

This deliberately bypasses a full data_pipeline.integrate/clean/reduce run,
same reasoning as cell_detect/finetune_gold.py: the raw Angelina source
folder ("data Angelina/books") does not exist on this machine at all (only
its already-trained checkpoint survives locally), and data_pipeline/crops/
(the Stage 2c crop archives braille_cnn_mixed.pt was actually trained on)
isn't present locally either. What *is* present locally is the raw DBSI
dataset ("data DBSI/"), so this mixes fresh gold crops with live DBSI crops
during training -- pure gold-only fine-tuning risks the same catastrophic
forgetting documented for the DBSI-only fine-tune in RESULTS.md (98.44% DBSI,
collapsed to ~14% on synthetic). Angelina can't be mixed in or regression-
checked on this machine for the same reason; that check should be redone
wherever data_pipeline/crops/ or the raw Angelina folder actually lives
(e.g. Colab) before trusting this checkpoint not to have regressed there.

Cell boxes are cropped with data_pipeline.transform.extract_crop using the
exact margins Stage 2 already reserved for each source (SOURCE_MARGINS in
transform.py: gold=0.15) and DBSI_MARGIN_SCALE from data_pipeline.integrate
(0.35, NOT DBSIDataset's own default of 0.8) -- so a DBSI crop here means the
same thing it meant when braille_cnn_mixed.pt was actually trained, and a
gold crop means what Stage 2 already intended it to mean once gold crops
existed.

Page split matches cell_detect/finetune_gold.py's gold split exactly, so no
page the detector fine-tune trained on becomes a held-out page here (or vice
versa): train=[1-8], val=[9,12], test=[10,11].

    py -3.11 -m braille_cnn.finetune_gold_local
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from data_pipeline.contracts import dot_string_to_code
from data_pipeline.crop_dataset import CropDataset
from data_pipeline.integrate import DBSI_MARGIN_SCALE
from data_pipeline.transform import IMG_SIZE_DEFAULT, extract_crop, margin_for

from .cnn import SimpleBrailleCNN
from .dbsi_dataset import DBSIDataset
from .eval_report import write_eval_report

ROOT = Path(__file__).resolve().parent.parent
GOLD_DIR = ROOT / "Gold Dataset" / "High quality dataset"
CROPS_DIR = ROOT / "data_pipeline" / "crops"
DEFAULT_DBSI_ROOT = ROOT / "data DBSI"
DEFAULT_INIT = ROOT / "braille_cnn" / "checkpoints" / "braille_cnn_mixed.pt"
DEFAULT_OUT = ROOT / "braille_cnn" / "checkpoints" / "braille_cnn_gold_finetuned.pt"

TRAIN_PAGES = [1, 2, 3, 4, 5, 6, 7, 8]
VAL_PAGES = [9, 12]
TEST_PAGES = [10, 11]

NUM_CLASSES = 64


def _build_gold_crops(pages: list[int], img_size: int = IMG_SIZE_DEFAULT):
    crops, codes, page_groups = [], [], []
    margin = margin_for("gold")
    for n in pages:
        json_path = GOLD_DIR / f"pg-{n}.json"
        img_path = next(
            (p for p in (GOLD_DIR / f"pg-{n}{ext}" for ext in (".jpeg", ".jpg", ".png")) if p.exists()),
            None,
        )
        if img_path is None or not json_path.exists():
            raise FileNotFoundError(f"Missing gold page data for pg-{n}")
        image = np.array(Image.open(img_path).convert("L"))
        doc = json.loads(json_path.read_text(encoding="utf-8"))
        for shape in doc.get("shapes", []):
            if shape.get("shape_type") != "rectangle":
                continue
            (px0, py0), (px1, py1) = shape["points"][:2]
            x0, x1 = sorted((float(px0), float(px1)))
            y0, y1 = sorted((float(py0), float(py1)))
            try:
                code = dot_string_to_code(shape.get("label", ""))
            except ValueError:
                continue
            crop = extract_crop(image, (x0, y0, x1, y1), margin=margin, img_size=img_size)
            if crop is None:
                continue
            crops.append(crop)
            codes.append(code)
            page_groups.append(f"gold:pg-{n}")
    crops_arr = np.stack(crops).astype(np.uint8) if crops else np.zeros((0, img_size, img_size), dtype=np.uint8)
    return {
        "crops": crops_arr,
        "codes": np.array(codes, dtype=np.int64),
        "sources": np.array(["gold"] * len(codes)),
        "page_groups": np.array(page_groups),
    }


def _save_npz(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


class _TensorLabelWrapper(torch.utils.data.Dataset):
    """CropDataset yields a plain python int label; DBSIDataset yields a 0-d
    tensor. ConcatDataset-ing them for one DataLoader batch makes collate
    choke on the mixed int/Tensor label type whenever a batch draws from
    both -- this normalizes CropDataset's label to match.
    """

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, label = self.base[idx]
        return image, torch.tensor(label, dtype=torch.long)


def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4c-gold -- fine-tune the classifier on real Gold crops")
    parser.add_argument("--dbsi-root", type=Path, default=DEFAULT_DBSI_ROOT)
    parser.add_argument("--init-checkpoint", type=Path, default=DEFAULT_INIT)
    parser.add_argument("--out-checkpoint", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device: {device}")

    print(f"Building gold crops -- train={TRAIN_PAGES} val={VAL_PAGES} test={TEST_PAGES}")
    gold_train_payload = _build_gold_crops(TRAIN_PAGES)
    gold_val_payload = _build_gold_crops(VAL_PAGES)
    gold_test_payload = _build_gold_crops(TEST_PAGES)
    print(f"  gold train: {len(gold_train_payload['codes'])} cells  "
          f"val: {len(gold_val_payload['codes'])}  test: {len(gold_test_payload['codes'])}")

    _save_npz(CROPS_DIR / "crops_gold_train.npz", gold_train_payload)
    _save_npz(CROPS_DIR / "crops_gold_val.npz", gold_val_payload)
    _save_npz(CROPS_DIR / "crops_gold_test.npz", gold_test_payload)

    gold_train_ds = CropDataset(CROPS_DIR / "crops_gold_train.npz", augment=True)
    gold_val_ds = CropDataset(CROPS_DIR / "crops_gold_val.npz", augment=False)
    gold_test_ds = CropDataset(CROPS_DIR / "crops_gold_test.npz", augment=False)
    print(f"gold train: {gold_train_ds.describe()}")
    print(f"gold val  : {gold_val_ds.describe()}")
    print(f"gold test : {gold_test_ds.describe()}")

    if not args.dbsi_root.exists():
        raise SystemExit(f"DBSI root not found: {args.dbsi_root}")
    print(f"loading DBSI train/test from {args.dbsi_root} (margin_scale={DBSI_MARGIN_SCALE}, "
          "matching the unified manifest, not DBSIDataset's own default 0.8)...")
    dbsi_train_ds = DBSIDataset(args.dbsi_root, split="train", margin_scale=DBSI_MARGIN_SCALE)
    dbsi_test_ds = DBSIDataset(args.dbsi_root, split="test", margin_scale=DBSI_MARGIN_SCALE)
    print(f"  DBSI train: {len(dbsi_train_ds)} cells  test: {len(dbsi_test_ds)} cells")

    train_ds = ConcatDataset([_TensorLabelWrapper(gold_train_ds), dbsi_train_ds])
    # Equal draw rate per domain regardless of size, so ~2k gold cells aren't
    # drowned out by ~20k DBSI cells (same rationale as
    # CropDataset.domain_balanced_sampler / finetune_angelina.py).
    weights = np.concatenate([
        np.full(len(gold_train_ds), 0.5 / max(len(gold_train_ds), 1)),
        np.full(len(dbsi_train_ds), 0.5 / max(len(dbsi_train_ds), 1)),
    ])
    sampler = WeightedRandomSampler(torch.from_numpy(weights).double(), num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0)

    gold_val_loader = DataLoader(gold_val_ds, batch_size=256, shuffle=False)
    rng = np.random.default_rng(0)
    dbsi_eval_idx = rng.choice(len(dbsi_test_ds), size=min(6000, len(dbsi_test_ds)), replace=False)
    from torch.utils.data import Subset

    dbsi_eval_loader = DataLoader(Subset(dbsi_test_ds, dbsi_eval_idx.tolist()), batch_size=256, shuffle=False)

    model = SimpleBrailleCNN(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(args.init_checkpoint, map_location=device, weights_only=True))
    print(f"loaded init checkpoint: {args.init_checkpoint}")

    def _snapshot():
        return evaluate(model, gold_val_loader, device), evaluate(model, dbsi_eval_loader, device)

    start_gold, start_dbsi = _snapshot()
    print(f"start -- gold val {start_gold:.4f}  DBSI (regression check) {start_dbsi:.4f}  "
          "Angelina: not checkable on this machine (no local Angelina data)")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    args.out_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    best_gold = start_gold
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = running_n = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            running_n += labels.size(0)
        gold_acc, dbsi_acc = _snapshot()
        print(f"epoch {epoch:02d}  loss={running_loss / max(running_n, 1):.4f}  "
              f"gold={gold_acc:.4f}  DBSI={dbsi_acc:.4f}")
        if gold_acc > best_gold:
            best_gold = gold_acc
            torch.save(model.state_dict(), args.out_checkpoint)
            print(f"  saved new best ({args.out_checkpoint})")

    if best_gold == start_gold:
        print("\nnever beat the starting gold-val accuracy; keeping the init checkpoint as final")
        model.load_state_dict(torch.load(args.init_checkpoint, map_location=device, weights_only=True))
        final_ckpt = args.init_checkpoint
    else:
        model.load_state_dict(torch.load(args.out_checkpoint, map_location=device, weights_only=True))
        final_ckpt = args.out_checkpoint
    final_gold, final_dbsi = _snapshot()

    print("\n=== held-out gold test (pg-[10, 11], ground-truth crops, never trained/monitored on) ===")
    test_loader = DataLoader(gold_test_ds, batch_size=256, shuffle=False)
    baseline_model = SimpleBrailleCNN(num_classes=NUM_CLASSES).to(device)
    baseline_model.load_state_dict(torch.load(args.init_checkpoint, map_location=device, weights_only=True))
    baseline_test_acc = evaluate(baseline_model, test_loader, device)
    tuned_test_acc = evaluate(model, test_loader, device)
    print(f"baseline ({args.init_checkpoint.name}): {baseline_test_acc:.4f}")
    print(f"fine-tuned ({final_ckpt.name}): {tuned_test_acc:.4f}")

    lines = [
        f"Train pages: pg-{TRAIN_PAGES} (mixed with live DBSI crops, equal draw rate) | "
        f"val (checkpoint selection only): pg-{VAL_PAGES} | "
        f"**test (held out, never trained/monitored on): pg-{TEST_PAGES}**",
        "",
        "Ground-truth crops (LabelMe boxes via data_pipeline.transform.extract_crop, "
        "margin=SOURCE_MARGINS['gold']=0.15), not detector output -- isolates classifier "
        "accuracy from cell-detection error.",
        "",
        "| model | gold test acc |",
        "|---|---|",
        f"| baseline (`{args.init_checkpoint.name}`) | {baseline_test_acc:.4f} |",
        f"| gold fine-tuned (`{final_ckpt.name}`) | {tuned_test_acc:.4f} |",
        "",
        f"Val-set snapshot, start -> best epoch: gold {start_gold:.4f} -> {final_gold:.4f} "
        f"(checkpoint-selection metric), DBSI {start_dbsi:.4f} -> {final_dbsi:.4f} "
        "(regression check only, not used for checkpoint selection, 6000-cell subset of DBSI test).",
        "",
        "Angelina accuracy could not be checked on this machine (raw Angelina data and the "
        "Stage 2c crops_*.npz archives braille_cnn_mixed.pt was actually trained on aren't "
        "present locally) -- re-check there before trusting this checkpoint on handheld-photo "
        "input.",
    ]
    write_eval_report(Path("reports/eval") / "gold_cnn_finetune.md",
                       "Gold CNN classifier fine-tune: before vs after (held-out test page)", lines)


if __name__ == "__main__":
    main()
