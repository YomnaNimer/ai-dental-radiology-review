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
    """Create a synthetic dental radiograph patch for model training."""
    base = NP_RNG.normal(168, 18, (size, size)).clip(70, 230)
    yy, xx = np.ogrid[:size, :size]

    # Tooth-like bright structure.
    cx = RNG.randint(10, 22)
    cy = RNG.randint(10, 22)
    rx = RNG.randint(9, 15)
    ry = RNG.randint(11, 17)
    tooth = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1
    base[tooth] += RNG.randint(20, 45)

    # Enamel border / anatomical variation.
    if RNG.random() < 0.45:
        base[:, RNG.randint(12, 20) : RNG.randint(21, 29)] += RNG.randint(8, 20)

    if has_lesion:
        lx = RNG.randint(10, 22)
        ly = RNG.randint(11, 25)
        radius = RNG.randint(4, 8)
        lesion = (xx - lx) ** 2 + (yy - ly) ** 2 <= radius**2
        base[lesion] -= RNG.randint(50, 80)
    elif RNG.random() < 0.35:
        # Non-lesion dark variation to make the task less trivial.
        lx = RNG.randint(5, 26)
        ly = RNG.randint(5, 26)
        radius = RNG.randint(2, 5)
        artifact = (xx - lx) ** 2 + (yy - ly) ** 2 <= radius**2
        base[artifact] -= RNG.randint(12, 28)

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
    for _ in range(420):
        patches.append(patch_features(make_patch(False)))
        labels.append(0)
        patches.append(patch_features(make_patch(True)))
        labels.append(1)

    model = RandomForestClassifier(n_estimators=140, max_depth=7, random_state=42)
    model.fit(np.vstack(patches), np.array(labels))
    return model


def create_demo_radiograph(width: int = 760, height: int = 420) -> tuple[Image.Image, list[tuple[int, int, int]]]:
    """Create a synthetic bitewing-like image with known suspicious dark areas."""
    img = Image.new("L", (width, height), 92)
    draw = ImageDraw.Draw(img)

    tooth_positions = [
        (75, 55, 165, 335),
        (150, 45, 250, 345),
        (235, 52, 345, 340),
        (330, 48, 440, 340),
        (425, 54, 535, 338),
        (520, 58, 625, 332),
        (610, 70, 720, 320),
    ]

    for box in tooth_positions:
        draw.rounded_rectangle(box, radius=38, fill=190, outline=224, width=5)
        x1, y1, x2, y2 = box
        root_mid = (x1 + x2) // 2
        draw.polygon([(x1 + 20, y2 - 20), (root_mid - 10, height - 30), (root_mid, y2 - 25)], fill=150)
        draw.polygon([(root_mid, y2 - 25), (root_mid + 10, height - 30), (x2 - 20, y2 - 20)], fill=145)

    # Alveolar bone and bitewing-style horizontal separation.
    draw.rectangle((0, 232, width, 250), fill=115)
    draw.line((0, 252, width, 238), fill=132, width=5)

    suspicious_regions = [
        (218, 198, 22),
        (388, 205, 25),
        (574, 187, 20),
    ]
    for x, y, r in suspicious_regions:
        draw.ellipse((x - r, y - r, x + r, y + r), fill=88)
        draw.ellipse((x - r // 2, y - r // 2, x + r // 2, y + r // 2), fill=72)

    img = img.filter(ImageFilter.GaussianBlur(radius=1.1))
    noise = NP_RNG.normal(0, 8, (height, width))
    arr = np.array(img).astype(float) + noise
    arr = arr.clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L"), suspicious_regions


def sliding_window_heatmap(image: Image.Image, model: RandomForestClassifier, patch_size: int = 32, step: int = 8) -> np.ndarray:
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


def plot_outputs(image: Image.Image, heatmap: np.ndarray, suspicious_regions: list[tuple[int, int, int]]) -> None:
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
    for x, y, r in suspicious_regions:
        circle = plt.Circle((x, y), r + 14, color="#19a39b", fill=False, linewidth=3)
        axes[2].add_patch(circle)
        axes[2].text(x + r + 10, y - r, "review", color="#19a39b", fontsize=10, weight="bold")

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
