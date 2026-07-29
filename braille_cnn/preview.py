import argparse

import matplotlib.pyplot as plt
import numpy as np

from .labels import code_to_label
from .render import render_braille_cell


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--out", type=str, default="braille_cnn/sample_grid.png")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(8, 8, figsize=(12, 12))
    for code in range(64):
        ax = axes[code // 8, code % 8]
        img = render_braille_cell(code, img_size=args.img_size, augment=not args.no_augment, rng=rng)
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(code_to_label(code), fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
