"""Temporary Gradio UI for manually checking the Sinhala Braille cell classifier.

The model (braille_cnn.cnn.SimpleBrailleCNN) classifies ONE Braille cell into
one of 64 dot patterns; output index == cell code (bit i set = dot i+1 raised).
The code is mapped to Sinhala with braille_cnn.labels.code_to_label(lang="si")
-- the same single-cell lookup the app uses. So "drawing a character" here
means drawing the cell's dots, not the Sinhala glyph.

Preprocessing matches braille_cnn.recognize._classify_boxes / infer_page:
grayscale -> bicubic resize to 64x64 -> normalize_crop -> (1, 1, 64, 64).

Run from the repo root:
    py -3.11 test_screen.py
"""

import sys
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from braille_cnn.cnn import SimpleBrailleCNN  # noqa: E402
from braille_cnn.labels import code_to_label  # noqa: E402
from braille_cnn.normalize import normalize_crop  # noqa: E402
from braille_cnn.render import DOT_GRID  # noqa: E402

CKPT_DIR = ROOT / "braille_cnn" / "checkpoints"
CHECKPOINTS = {
    # gold_finetuned is what export_onnx.py ships to the Flutter app
    "braille_cnn_gold_finetuned.pt (app model)": CKPT_DIR / "braille_cnn_gold_finetuned.pt",
    "braille_cnn_mixed.pt": CKPT_DIR / "braille_cnn_mixed.pt",
}
NUM_CLASSES = 64
IMG_SIZE = 64
CANVAS = 320
DEVICE = torch.device("cpu")

_models: dict[str, SimpleBrailleCNN] = {}


def get_model(name: str) -> SimpleBrailleCNN:
    if name not in _models:
        model = SimpleBrailleCNN(num_classes=NUM_CLASSES)
        model.load_state_dict(torch.load(CHECKPOINTS[name], map_location=DEVICE, weights_only=True))
        model.eval()
        _models[name] = model
    return _models[name]


def dots_of(code: int) -> str:
    return "-".join(str(d) for d in range(1, 7) if code & (1 << (d - 1))) or "none"


def class_name(code: int) -> str:
    char = code_to_label(code, lang="si") if code else "(blank)"
    return f"{char}   code {code} · dots {dots_of(code)}"


def dot_guide() -> Image.Image:
    """Faint rings where the training renderer puts dots 1-6 (render._dot_centers,
    cell_scale=1), so drawn dots land in the positions the model expects."""
    img = Image.new("RGBA", (CANVAS, CANVAS), (255, 255, 255, 255))
    d = ImageDraw.Draw(img)
    r = 0.085 * CANVAS
    for dot, (row, col) in DOT_GRID.items():
        x = CANVAS / 2 + (col - 0.5) * 0.30 * CANVAS
        y = CANVAS / 2 + (row - 1) * 0.28 * CANVAS
        d.ellipse((x - r, y - r, x + r, y + r), outline=(200, 200, 200, 255), width=2)
        d.text((x - 4, y - 7), str(dot), fill=(200, 200, 200, 255))
    return img


def to_gray(img: Image.Image) -> Image.Image:
    """Flatten any transparency onto white, then grayscale."""
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return img.convert("L")


def preprocess(img: Image.Image, invert: bool) -> torch.Tensor:
    gray = to_gray(img)
    if invert:
        gray = Image.eval(gray, lambda v: 255 - v)
    gray = gray.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BICUBIC)
    arr = normalize_crop(gray)  # float32 [0,1], mean-centred at 0.5
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, 64, 64)


def predict(img: Image.Image | None, model_name: str, invert: bool):
    if img is None:
        return {}, None
    x = preprocess(img, invert)
    with torch.no_grad():
        probs = torch.softmax(get_model(model_name)(x), dim=1)[0]
    top = torch.topk(probs, 5)
    label = {class_name(int(i)): float(p) for p, i in zip(top.values, top.indices)}
    model_view = Image.fromarray((x[0, 0].numpy() * 255).astype(np.uint8)).resize((192, 192), Image.NEAREST)
    return label, model_view


def sketch_to_image(value) -> Image.Image | None:
    """Use only the drawn layers (not the guide background) on white."""
    if not value:
        return None
    layers = [l for l in (value.get("layers") or []) if l is not None]
    if not layers:
        return None
    out = Image.new("RGBA", (CANVAS, CANVAS), (255, 255, 255, 255))
    for layer in layers:
        layer = Image.fromarray(layer) if isinstance(layer, np.ndarray) else layer
        out = Image.alpha_composite(out, layer.convert("RGBA").resize(out.size))
    if np.asarray(out.convert("L")).min() > 250:  # nothing drawn yet
        return None
    return out


def predict_sketch(value, model_name, invert):
    return predict(sketch_to_image(value), model_name, invert)


with gr.Blocks(title="Sinhala Braille cell classifier — manual test") as demo:
    gr.Markdown(
        "## Sinhala Braille cell classifier — manual test\n"
        "Draw the **dots of one Braille cell** (dark on white, one tap per dot inside the grey rings), "
        "or upload a cropped single-cell image. Dot numbering: `1 4 / 2 5 / 3 6`."
    )
    with gr.Row():
        model_dd = gr.Dropdown(list(CHECKPOINTS), value=next(iter(CHECKPOINTS)), label="Checkpoint")
        invert_cb = gr.Checkbox(False, label="Invert input (use for light dots on dark background)")
    with gr.Row():
        with gr.Column():
            with gr.Tab("Draw"):
                sketch = gr.Sketchpad(
                    value={"background": dot_guide(), "layers": [], "composite": None},
                    type="pil",
                    image_mode="RGBA",
                    canvas_size=(CANVAS, CANVAS),
                    brush=gr.Brush(default_size=40, colors=["#000000"], color_mode="fixed"),
                    label="Draw cell dots",
                )
                draw_btn = gr.Button("Predict drawing", variant="primary")
            with gr.Tab("Upload"):
                upload = gr.Image(type="pil", label="Single cropped Braille cell")
                upload_btn = gr.Button("Predict upload", variant="primary")
        with gr.Column():
            result = gr.Label(num_top_classes=5, label="Top-5 predictions")
            model_view = gr.Image(label="What the model sees (64×64 after normalize_crop)", interactive=False)

    draw_btn.click(predict_sketch, [sketch, model_dd, invert_cb], [result, model_view])
    upload_btn.click(predict, [upload, model_dd, invert_cb], [result, model_view])
    upload.change(predict, [upload, model_dd, invert_cb], [result, model_view])

if __name__ == "__main__":
    demo.launch(share=False)
