from pathlib import Path
from random import Random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from sklearn.ensemble import RandomForestClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "figures"
DATA_DIR = PROJECT_ROOT / "data"
RNG = Random(42)
NP_RNG = np.random.default_rng(42)


def make_patch(has_lesion: bool, size: int = 32) -> np.ndarray:
    """Create a synthetic interproximal bitewing patch for model training."""
    base = NP_RNG.normal(150, 14, (size, size)).clip(60, 220)
    yy, xx = np.ogrid[:size, :size]

    # Two adjacent proximal enamel surfaces, leaving a narrow contact/interproximal area.
    left_tooth = np.broadcast_to(xx < RNG.randint(13, 16), base.shape)
    right_tooth = np.broadcast_to(xx > RNG.randint(17, 20), base.shape)
    base[left_tooth] += RNG.randint(25, 45)
    base[right_tooth] += RNG.randint(25, 45)

    # Brighter enamel edges and slightly darker dentin-like center.
    base[:, 12:15] += RNG.randint(20, 38)
    base[:, 17:20] += RNG.randint(20, 38)
    base[8:26, :10] -= RNG.randint(8, 18)
    base[8:26, 22:] -= RNG.randint(8, 18)

    if has_lesion:
        # Interproximal/class II-style triangular radiolucency at the contact area.
        side = RNG.choice(["left", "right"])
        y0 = RNG.randint(10, 17)
        height = RNG.randint(9, 14)
        if side == "left":
            lesion = (
                (xx >= 10)
                & (xx <= 17)
                & (yy >= y0)
                & (yy <= y0 + height)
                & ((xx - 10) <= (yy - y0) * 0.75 + 1)
            )
        else:
            lesion = (
                (xx <= 22)
                & (xx >= 15)
                & (yy >= y0)
                & (yy <= y0 + height)
                & ((22 - xx) <= (yy - y0) * 0.75 + 1)
            )
        base[lesion] -= RNG.randint(55, 85)
    elif RNG.random() < 0.35:
        # Non-lesion cervical/interproximal burnout-like variation to make the task less trivial.
        lx = RNG.randint(5, 26)
        ly = RNG.randint(18, 28)
        radius = RNG.randint(2, 5)
        artifact = (xx - lx) ** 2 + (yy - ly) ** 2 <= radius**2
        base[artifact] -= RNG.randint(10, 24)

    return base.clip(0, 255).astype(np.uint8)


def patch_features(patch: np.ndarray) -> np.ndarray:
    """Extract simple image features from a patch."""
    patch = patch.astype(float)
    center = patch[8:24, 8:24]
    gy, gx = np.gradient(patch)
    dark_threshold = np.percentile(patch, 25)
    return np.array(
        [
            patch.mean(),
            patch.std(),
            center.mean(),
            center.std(),
            (patch < dark_threshold).mean(),
            (center < dark_threshold).mean(),
            np.abs(gx).mean() + np.abs(gy).mean(),
            np.percentile(patch, 90) - np.percentile(patch, 10),
        ]
    )


def train_patch_classifier() -> RandomForestClassifier:
    """Train a toy classifier on synthetic normal and radiolucent patches."""
    patches = []
    labels = []
    for _ in range(180):
        patches.append(patch_features(make_patch(False)))
        labels.append(0)
        patches.append(patch_features(make_patch(True)))
        labels.append(1)

    model = RandomForestClassifier(n_estimators=60, max_depth=7, random_state=42, n_jobs=1)
    model.fit(np.vstack(patches), np.array(labels))
    return model


def rounded_tooth(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], pulp: bool = True) -> None:
    """Draw a posterior crown with enamel rim, dentin body, and pulp chamber."""
    x1, y1, x2, y2 = box
    draw.rounded_rectangle(box, radius=18, fill=178, outline=224, width=5)
    inset = 9
    draw.rounded_rectangle((x1 + inset, y1 + inset, x2 - inset, y2 - inset), radius=14, fill=158)
    # Occlusal enamel cap and fissure-like anatomy.
    draw.rounded_rectangle((x1 + 8, y1 + 6, x2 - 8, y1 + 24), radius=10, fill=210)
    draw.arc((x1 + 16, y1 + 18, x2 - 16, y1 + 48), 200, 340, fill=118, width=2)
    if pulp:
        px = (x1 + x2) // 2
        py = y1 + int((y2 - y1) * 0.58)
        draw.ellipse((px - 7, py - 4, px + 7, py + 4), fill=118)


def draw_interproximal_lesion(
    draw: ImageDraw.ImageDraw,
    contact_x: int,
    contact_y: int,
    direction: str,
    depth: int = 28,
) -> list[tuple[int, int]]:
    """Draw a triangular proximal/class II-style radiolucency near a contact point."""
    if direction == "left":
        points = [
            (contact_x - 2, contact_y - 13),
            (contact_x - 2, contact_y + 15),
            (contact_x - depth, contact_y + 4),
        ]
    else:
        points = [
            (contact_x + 2, contact_y - 13),
            (contact_x + 2, contact_y + 15),
            (contact_x + depth, contact_y + 4),
        ]
    draw.polygon(points, fill=48)
    return points


def create_demo_radiograph(width: int = 860, height: int = 520) -> tuple[Image.Image, list[list[tuple[int, int]]]]:
    """Create a more bitewing-like posterior image with interproximal caries."""
    img = Image.new("L", (width, height), 82)
    draw = ImageDraw.Draw(img)

    upper_teeth = [
        (58, 42, 145, 214),
        (138, 34, 238, 222),
        (228, 36, 338, 224),
        (328, 40, 440, 220),
        (430, 48, 540, 216),
        (532, 58, 634, 208),
        (626, 70, 728, 198),
    ]
    lower_teeth = [
        (70, 300, 160, 472),
        (152, 288, 254, 482),
        (244, 286, 356, 486),
        (346, 292, 458, 480),
        (448, 300, 558, 470),
        (550, 310, 654, 458),
        (646, 320, 748, 450),
    ]

    for box in upper_teeth + lower_teeth:
        rounded_tooth(draw, box)

    # Bitewing features: occlusal gap, alveolar crest, and slight projection overlap.
    draw.rectangle((0, 238, width, 282), fill=58)
    draw.line((24, 230, width - 24, 236), fill=134, width=6)
    draw.line((28, 290, width - 28, 284), fill=132, width=6)
    for x in [145, 238, 338, 440, 540, 634]:
        draw.rectangle((x - 5, 70, x + 5, 215), fill=168)
    for x in [160, 254, 356, 458, 558, 654]:
        draw.rectangle((x - 5, 306, x + 5, 468), fill=165)

    suspicious_regions = [
        draw_interproximal_lesion(draw, 238, 160, "left", 30),
        draw_interproximal_lesion(draw, 440, 162, "right", 30),
        draw_interproximal_lesion(draw, 356, 380, "left", 32),
        draw_interproximal_lesion(draw, 558, 374, "right", 28),
    ]

    img = img.filter(ImageFilter.GaussianBlur(radius=1.1))
    noise = NP_RNG.normal(0, 8, (height, width))
    arr = np.array(img).astype(float) + noise
    arr = arr.clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L"), suspicious_regions


def sliding_window_heatmap(image: Image.Image, model: RandomForestClassifier, patch_size: int = 32, step: int = 12) -> np.ndarray:
    arr = np.array(image)
    heatmap = np.zeros_like(arr, dtype=float)
    counts = np.zeros_like(arr, dtype=float)

    for y in range(0, arr.shape[0] - patch_size + 1, step):
        for x in range(0, arr.shape[1] - patch_size + 1, step):
            patch = arr[y : y + patch_size, x : x + patch_size]
            probability = model.predict_proba([patch_features(patch)])[0, 1]
            heatmap[y : y + patch_size, x : x + patch_size] += probability
            counts[y : y + patch_size, x : x + patch_size] += 1

    heatmap = np.divide(heatmap, counts, out=np.zeros_like(heatmap), where=counts > 0)
    return heatmap


def plot_outputs(image: Image.Image, heatmap: np.ndarray, suspicious_regions: list[list[tuple[int, int]]]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    image.save(DATA_DIR / "synthetic_bitewing_example.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Synthetic bitewing-style example")
    axes[0].axis("off")

    axes[1].imshow(image, cmap="gray")
    axes[1].imshow(heatmap, cmap="magma", alpha=0.58, vmin=0, vmax=1)
    axes[1].set_title("AI probability heatmap")
    axes[1].axis("off")

    axes[2].imshow(image, cmap="gray")
    axes[2].set_title("Flagged regions for dentist review")
    axes[2].axis("off")
    for polygon in suspicious_regions:
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        pad = 18
        x0, x1 = min(xs) - pad, max(xs) + pad
        y0, y1 = min(ys) - pad, max(ys) + pad
        rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, color="#19a39b", fill=False, linewidth=3)
        axes[2].add_patch(rect)
        axes[2].text(x1 + 5, y0 + 9, "interproximal\nreview", color="#19a39b", fontsize=9, weight="bold")

    fig.suptitle(
        "Educational AI-assisted dental radiology workflow: detection support, not diagnosis",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.02,
        "Note: synthetic image and toy classifier for portfolio demonstration only. Not for clinical diagnosis.",
        fontsize=9,
        color="#606c76",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.savefig(FIGURES_DIR / "ai_radiology_demo_workflow.png", dpi=220)
    plt.close(fig)


def main() -> None:
    model = train_patch_classifier()
    image, suspicious_regions = create_demo_radiograph()
    heatmap = sliding_window_heatmap(image, model)
    plot_outputs(image, heatmap, suspicious_regions)
    print("Saved practical AI demo outputs:")
    print(f"- {DATA_DIR / 'synthetic_bitewing_example.png'}")
    print(f"- {FIGURES_DIR / 'ai_radiology_demo_workflow.png'}")


if __name__ == "__main__":
    main()
