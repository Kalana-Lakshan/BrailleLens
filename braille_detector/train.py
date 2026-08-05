"""Trains BrailleDetector on combined DBSI + Angelina page data.

Scoped down from the paper's full training recipe (500 epochs, full
resize/stretch/rotation augmentation, lambda_cls annealed 1->1000) to
something that finishes in a reasonable time on this machine as a first
working proof-of-concept -- see README.md in this folder for what's
scoped down and why, and what a fuller run would need.
"""

import argparse
from pathlib import Path

import torch
from torch import optim
from torch.utils.data import DataLoader

from .data import BraillePageDataset, collate_fn, list_angelina_pages, list_dbsi_pages
from .loss import compute_loss
from .model import BrailleDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--crop-size", type=int, default=416)
    parser.add_argument("--lambda-cls", type=float, default=50.0)
    parser.add_argument("--out", type=str, default="braille_detector/checkpoints/detector.pt")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--init-checkpoint", type=str, default=None,
                         help="resume training from this checkpoint's weights instead of random init")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    dbsi_pages = list_dbsi_pages()
    angelina_pages = list_angelina_pages(split="train")
    pages = dbsi_pages + angelina_pages
    print(f"DBSI pages: {len(dbsi_pages)}  Angelina pages: {len(angelina_pages)}  total: {len(pages)}")

    dataset = BraillePageDataset(pages, crop_size=args.crop_size, epoch_length=args.steps * args.batch_size, train=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    model = BrailleDetector().to(device)
    if args.init_checkpoint:
        model.load_state_dict(torch.load(args.init_checkpoint, map_location=device))
        print(f"resumed weights from {args.init_checkpoint}")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model.train()
    running = {"total": 0.0, "box": 0.0, "cls": 0.0, "pos": 0.0}
    step = 0
    for images, boxes_list, labels_list in loader:
        images = images.to(device)
        box_out, cls_out = model(images)
        loss, box_loss, cls_loss, num_pos = compute_loss(
            model, box_out, cls_out, boxes_list, labels_list, lambda_cls=args.lambda_cls
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running["total"] += loss.item()
        running["box"] += box_loss.item()
        running["cls"] += cls_loss.item()
        running["pos"] += num_pos
        step += 1

        if step % args.log_every == 0:
            n = args.log_every
            print(f"step {step:5d}  loss {running['total']/n:.4f}  box {running['box']/n:.4f}  "
                  f"cls {running['cls']/n:.4f}  avg_pos/img {running['pos']/n/args.batch_size:.1f}")
            running = {"total": 0.0, "box": 0.0, "cls": 0.0, "pos": 0.0}

        if step % args.save_every == 0:
            torch.save(model.state_dict(), out_path)
            print(f"  saved checkpoint to {out_path}")

        if step >= args.steps:
            break

    torch.save(model.state_dict(), out_path)
    print(f"final checkpoint saved to {out_path}")


if __name__ == "__main__":
    main()
