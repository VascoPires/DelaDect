"""Synthetic robustness study: operating range and parameter-scale sensitivity.

Companion to ``examples/synthetic_delamination_validation.py``. That example
is one seed at one fixed contrast -- useful to verify correctness, but not
to show *where* detection starts to fail. This script instead:

1. Sweeps diffuse-delamination contrast (``alpha_d``) against diffuse-region
   width (relative to ``avg_crack_width_px``) over multiple noise
   realizations, and reports median final-state IoU as a heatmap -- the
   closest DelaDect equivalent to a MatrixCraCS-style detectability map.
2. Does the same for edge delamination: contrast vs. image noise.
3. Shows, on one representative case, what ``window_diffuse`` actually does
   across four settings (too small / appropriate / too large / much too
   large), plus a direct package-default-vs-scale-matched comparison on the
   main example's real geometry -- so the choice in the main example reads
   as a documented trade-off, not an unexplained tuning knob.

To keep a ~250-cell parameter sweep tractable, grid cells 1 and 2 skip raw
8-bit image rendering and preprocessing entirely and instead synthesize the
*ratio-normalized processed frame* directly (uniform background = 255,
damage darkened by the swept contrast, plus Gaussian noise in ratio space).
This is what ``preprocess_stack_to_disk(reference_mode="static")`` would
produce from a real raw frame pair in the noise-free-background limit, and
isolates the detection threshold behaviour itself from preprocessing
effects -- it is not a claim that raw-image rendering is unnecessary in
general (the main example still does full raw-image rendering). Section 3's
package-default-vs-scale-matched comparison instead reuses the main
example's real generated sequence and real preprocessing, since that
comparison is specifically about the main demonstration's own geometry.

Regenerate with::

    python examples/synthetic_delamination_robustness.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from synthetic_delamination_validation import (  # noqa: E402
    DETECTION_DIFFUSE_PARAMS,
    _configure_style,
    _wavy_ellipse_mask,
    generate_synthetic_sequence,
    iou_score,
    make_standalone_owner,
)

DOCS_STATIC_DIR = REPO_ROOT / "docs" / "source" / "_static" / "delamination"
RESULTS_ROOT = REPO_ROOT / "results" / "synthetic_validation" / "robustness"

# ---------------------------------------------------------------------------
# Shared small-scale geometry (grid cells 1 and 2)
# ---------------------------------------------------------------------------

SWEEP_SHAPE = (160, 240)
SWEEP_AVG_CRACK_WIDTH_PX = 8.0
SWEEP_CRACK = np.array([[80.0, 90.0], [80.0, 150.0]]).reshape(1, 2, 2)  # (y, x) endpoints
SWEEP_CRACK_CENTER = (80.0, 120.0)

# ---------------------------------------------------------------------------
# 1. Diffuse detectability map: contrast x width -> median final-frame IoU
# ---------------------------------------------------------------------------

DIFFUSE_CONTRAST_VALUES: Tuple[float, ...] = (0.03, 0.05, 0.08, 0.12, 0.18)
DIFFUSE_WIDTH_SCALES: Tuple[float, ...] = (1, 2, 3, 4, 6)
DIFFUSE_SWEEP_SEEDS: range = range(10)
DIFFUSE_SWEEP_NOISE_STD = 3.0  # fixed; only contrast and width are swept here


def _diffuse_case(
    alpha_d: float, width_scale: float, seed: int, *, return_arrays: bool = False,
) -> Any:
    """Run one (contrast, width, seed) diffuse-detectability cell; return IoU.

    With ``return_arrays=True``, instead returns
    ``{"frame": frame1_u8, "gt_mask": mask, "detected": detected, "iou": iou}``
    for illustration (see :func:`make_sweep_example_figure`).
    """
    from deladect.detection.delamination.diffuse import DiffuseDetector

    rng = np.random.default_rng(seed)
    radius = 0.5 * width_scale * SWEEP_AVG_CRACK_WIDTH_PX
    mask = _wavy_ellipse_mask(
        SWEEP_SHAPE, SWEEP_CRACK_CENTER[0], SWEEP_CRACK_CENTER[1], radius, radius,
        wave_amp=0.10, wave_freq=7.0,
    )

    frame0 = np.full(SWEEP_SHAPE, 255.0, dtype=np.float64)
    frame1 = 255.0 * (1.0 - alpha_d * mask) + rng.normal(0.0, DIFFUSE_SWEEP_NOISE_STD, SWEEP_SHAPE)
    frame0_u8 = frame0.astype(np.uint8)
    frame1_u8 = np.clip(frame1, 0.0, 255.0).astype(np.uint8)

    owner = make_standalone_owner(SWEEP_AVG_CRACK_WIDTH_PX)
    result = DiffuseDetector(owner).diffuse_delamination(
        cracks=[np.zeros((0, 2, 2)), SWEEP_CRACK],
        processed_stack=[frame0_u8, frame1_u8],
        params=DETECTION_DIFFUSE_PARAMS,
        save_overlays=False,
    )
    detected = np.asarray(result["masks"]["frame_0001"], dtype=bool)
    iou = iou_score(detected, mask)
    if return_arrays:
        return {"frame": frame1_u8, "gt_mask": mask, "detected": detected, "iou": iou}
    return iou


def run_diffuse_detectability_sweep(
    *,
    contrast_values: Tuple[float, ...] = DIFFUSE_CONTRAST_VALUES,
    width_scales: Tuple[float, ...] = DIFFUSE_WIDTH_SCALES,
    seeds: range = DIFFUSE_SWEEP_SEEDS,
) -> pd.DataFrame:
    """Run the full diffuse detectability grid; one row per (contrast, width, seed)."""
    rows: List[Dict[str, Any]] = []
    for alpha_d in contrast_values:
        for width_scale in width_scales:
            for seed in seeds:
                iou = _diffuse_case(alpha_d, width_scale, seed)
                rows.append({
                    "alpha_d": alpha_d, "width_scale": width_scale, "seed": seed, "iou": iou,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Edge detectability map: contrast x image noise -> median final-frame IoU
# ---------------------------------------------------------------------------

EDGE_CONTRAST_VALUES: Tuple[float, ...] = (0.08, 0.11, 0.15, 0.22)
EDGE_NOISE_STD_VALUES: Tuple[float, ...] = (2.0, 10.0, 20.0, 30.0)
EDGE_SWEEP_SEEDS: range = range(10)
EDGE_FRONT_HEIGHT_PX = 30  # fixed; only contrast and noise are swept here
EDGE_SWEEP_PARAMS: Dict[str, Any] = {
    "window_edge": (1, 20),
    "gaussian_filters": (0.5, 15.0),
    "hard_floor": 0.90,
    "scale_min_percentile": 10,
    "scale_max_percentile": 95,
    "seed_ratio": 0.01,
    "post_threshold_closing_px": 6,
}


def _edge_case(
    alpha_e: float, noise_std: float, seed: int, *, return_arrays: bool = False,
) -> Any:
    """Run one (contrast, noise, seed) edge-detectability cell; return IoU.

    With ``return_arrays=True``, instead returns
    ``{"frame": frame1_u8, "gt_mask": gt, "detected": detected, "iou": iou}``.
    """
    from deladect.detection.delamination.edge import EdgeDetector

    rng = np.random.default_rng(seed)
    gt = np.zeros(SWEEP_SHAPE, dtype=bool)
    gt[:EDGE_FRONT_HEIGHT_PX, :] = True

    frame0 = np.full(SWEEP_SHAPE, 255.0, dtype=np.float64)
    frame1 = frame0.copy()
    frame1[:EDGE_FRONT_HEIGHT_PX, :] *= (1.0 - alpha_e)
    frame1 = frame1 + rng.normal(0.0, noise_std, SWEEP_SHAPE)
    frame0_u8 = frame0.astype(np.uint8)
    frame1_u8 = np.clip(frame1, 0.0, 255.0).astype(np.uint8)

    owner = make_standalone_owner(SWEEP_AVG_CRACK_WIDTH_PX)
    result = EdgeDetector(owner).detect_primary(
        processed_stack=[frame0_u8, frame1_u8], params=EDGE_SWEEP_PARAMS, save_overlays=False,
    )
    detected = np.asarray(result["masks"]["frame_0001"], dtype=bool)
    iou = iou_score(detected, gt)
    if return_arrays:
        return {"frame": frame1_u8, "gt_mask": gt, "detected": detected, "iou": iou}
    return iou


def run_edge_detectability_sweep(
    *,
    contrast_values: Tuple[float, ...] = EDGE_CONTRAST_VALUES,
    noise_values: Tuple[float, ...] = EDGE_NOISE_STD_VALUES,
    seeds: range = EDGE_SWEEP_SEEDS,
) -> pd.DataFrame:
    """Run the full edge detectability grid; one row per (contrast, noise, seed)."""
    rows: List[Dict[str, Any]] = []
    for alpha_e in contrast_values:
        for noise_std in noise_values:
            for seed in seeds:
                iou = _edge_case(alpha_e, noise_std, seed)
                rows.append({
                    "alpha_e": alpha_e, "noise_std": noise_std, "seed": seed, "iou": iou,
                })
    return pd.DataFrame(rows)


def _median_iou_grid(df: pd.DataFrame, row_col: str, col_col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pivot a sweep DataFrame to a (row_values, col_values, median_iou_grid)."""
    pivot = df.pivot_table(index=row_col, columns=col_col, values="iou", aggfunc="median")
    pivot = pivot.sort_index().sort_index(axis=1)
    return pivot.index.to_numpy(), pivot.columns.to_numpy(), pivot.to_numpy()


def _plot_heatmap(
    ax: Any,
    row_values: np.ndarray,
    col_values: np.ndarray,
    grid: np.ndarray,
    *,
    row_label: str,
    col_label: str,
    title: str,
) -> None:
    heatmap_label_fontsize = 11.5
    heatmap_tick_fontsize = 9.5
    heatmap_title_fontsize = 13.0
    heatmap_value_fontsize = 9.0

    im = ax.imshow(grid, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto", origin="lower")
    ax.set_xticks(range(len(col_values)))
    ax.set_xticklabels([str(v) for v in col_values])
    ax.set_yticks(range(len(row_values)))
    ax.set_yticklabels([f"{v:g}" for v in row_values])
    ax.set_xlabel(col_label, fontsize=heatmap_label_fontsize)
    ax.set_ylabel(row_label, fontsize=heatmap_label_fontsize)
    ax.set_title(title, fontsize=heatmap_title_fontsize)
    ax.tick_params(axis="both", labelsize=heatmap_tick_fontsize)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            value = grid[i, j]
            color = "white" if value < 0.55 else "black"
            ax.text(
                j, i, f"{value:.2f}", ha="center", va="center",
                fontsize=heatmap_value_fontsize, color=color,
            )
    return im


def make_detectability_figure(
    diffuse_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    *,
    output_dir: Path,
) -> Path:
    import matplotlib.pyplot as plt

    _configure_style()

    widths, contrasts_d, grid_d = _median_iou_grid(diffuse_df, "alpha_d", "width_scale")
    noises, contrasts_e, grid_e = _median_iou_grid(edge_df, "alpha_e", "noise_std")

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6))
    im0 = _plot_heatmap(
        axes[0], widths, contrasts_d, grid_d,
        row_label=r"Diffuse contrast $\alpha_d$",
        col_label="Width scale (x avg_crack_width_px)",
        title="(a) Diffuse detectability",
    )
    im1 = _plot_heatmap(
        axes[1], noises, contrasts_e, grid_e,
        row_label=r"Edge contrast $\alpha_e$",
        col_label="Noise std [8-bit levels]",
        title="(b) Edge detectability",
    )
    colorbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    colorbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    for colorbar in (colorbar0, colorbar1):
        colorbar.set_label("Median IoU", fontsize=11.5)
        colorbar.ax.tick_params(labelsize=9.5)
    fig.suptitle(
        f"Detectability map (median IoU over {len(DIFFUSE_SWEEP_SEEDS)} noise realizations per cell)",
        fontsize=14,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "synthetic_validation_detectability.png"
    pdf_path = output_dir / "synthetic_validation_detectability.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path


# ---------------------------------------------------------------------------
# 1b. What the sweep test images actually look like (illustration only --
# not part of the sweep itself)
# ---------------------------------------------------------------------------

# A representative subset of each grid, not the full grid: enough to show
# the range without an 8-column figure.
DIFFUSE_EXAMPLE_WIDTH_SCALES: Tuple[float, ...] = (1, 2, 3, 4, 6)
DIFFUSE_EXAMPLE_ALPHA_D = 0.12
EDGE_EXAMPLE_CONTRAST_VALUES: Tuple[float, ...] = (0.08, 0.11, 0.12, 0.15, 0.22)
EDGE_EXAMPLE_NOISE_STD = 10.0
EXAMPLE_SEED = 0

# The illustration figures are embedded at documentation-column width, where
# Matplotlib's compact defaults become difficult to read.  Keep every visible
# text element at least 80% larger than in the original figures.
SWEEP_EXAMPLE_FONT_SCALE = 2.1
WINDOW_FIGURE_FONT_SCALE = 1.8


def make_sweep_example_figure(*, output_dir: Path) -> Path:
    """Show actual example test images from the detectability sweeps.

    The sweep grids themselves (250 + 160 cells) are numbers, not pictures.
    This renders a handful of the actual tiny synthetic patches used to
    produce them -- same size (160x240), same one-crack/one-blob geometry,
    same flat background, no texture -- so the sweep's methodology is
    visible rather than only described.
    """
    import matplotlib.pyplot as plt

    _configure_style()

    fig, axes = plt.subplots(2, 6, figsize=(15.0, 6.6))

    for ax in axes[0, 5:]:
        ax.axis("off")

    for col, width_scale in enumerate(DIFFUSE_EXAMPLE_WIDTH_SCALES):
        case = _diffuse_case(DIFFUSE_EXAMPLE_ALPHA_D, width_scale, EXAMPLE_SEED, return_arrays=True)
        ax = axes[0, col]
        ax.imshow(case["frame"], cmap="gray", vmin=0, vmax=255)
        overlay = np.zeros((*case["detected"].shape, 4), dtype=np.float32)
        overlay[case["detected"], 1] = 0.7
        overlay[case["detected"], 3] = 0.5
        ax.imshow(overlay)
        ax.contour(case["gt_mask"], levels=[0.5], colors="black", linewidths=0.7)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"width={width_scale:g}x\nIoU={case['iou']:.2f}",
            fontsize=8 * SWEEP_EXAMPLE_FONT_SCALE,
        )
    axes[0, 0].set_ylabel(
        "Diffuse\n" + r"($\alpha_d$=" + f"{DIFFUSE_EXAMPLE_ALPHA_D:g})",
        fontsize=9 * SWEEP_EXAMPLE_FONT_SCALE,
    )

    for col, alpha_e in enumerate(EDGE_EXAMPLE_CONTRAST_VALUES):
        case = _edge_case(alpha_e, EDGE_EXAMPLE_NOISE_STD, EXAMPLE_SEED, return_arrays=True)
        ax = axes[1, col]
        ax.imshow(case["frame"], cmap="gray", vmin=0, vmax=255)
        overlay = np.zeros((*case["detected"].shape, 4), dtype=np.float32)
        overlay[case["detected"], 0] = 0.85
        overlay[case["detected"], 3] = 0.4
        ax.imshow(overlay)
        ax.contour(case["gt_mask"], levels=[0.5], colors="black", linewidths=0.7)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            rf"$\alpha_e$={alpha_e:g}" + f"\nIoU={case['iou']:.2f}",
            fontsize=8 * SWEEP_EXAMPLE_FONT_SCALE,
        )
    axes[1, 5].axis("off")
    axes[1, 0].set_ylabel(
        "Edge\n" + f"(noise std={EDGE_EXAMPLE_NOISE_STD:g})",
        fontsize=9 * SWEEP_EXAMPLE_FONT_SCALE,
    )

    fig.suptitle(
        "What the detectability sweep's test images actually look like\n"
        f"(one seed shown per cell; the sweep itself uses {len(DIFFUSE_SWEEP_SEEDS)} per cell)",
        fontsize=10 * SWEEP_EXAMPLE_FONT_SCALE,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.84))

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "synthetic_validation_sweep_examples.png"
    pdf_path = output_dir / "synthetic_validation_sweep_examples.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path


# ---------------------------------------------------------------------------
# 3. window_diffuse: a worked parameter-scale lesson
# ---------------------------------------------------------------------------

WINDOW_DIFFUSE_DEMO_ALPHA_D = 0.18
WINDOW_DIFFUSE_DEMO_WIDTH_SCALE = 6.0
WINDOW_DIFFUSE_DEMO_SEED = 0
# The last setting is exactly the package default (0, 60), tying this panel
# directly to the default-vs-scale-matched comparison computed below.
WINDOW_DIFFUSE_VARIANTS: Tuple[Tuple[int, int], ...] = ((1, 10), (1, 20), (1, 40), (0, 60))


def run_window_diffuse_demo() -> Dict[str, Any]:
    """Run the same diffuse blob through four ``window_diffuse`` settings."""
    from deladect.detection.delamination.diffuse import DiffuseDetector

    rng = np.random.default_rng(WINDOW_DIFFUSE_DEMO_SEED)
    radius = 0.5 * WINDOW_DIFFUSE_DEMO_WIDTH_SCALE * SWEEP_AVG_CRACK_WIDTH_PX
    mask = _wavy_ellipse_mask(
        SWEEP_SHAPE, SWEEP_CRACK_CENTER[0], SWEEP_CRACK_CENTER[1], radius, radius,
        wave_amp=0.10, wave_freq=7.0,
    )
    frame0 = np.full(SWEEP_SHAPE, 255.0, dtype=np.float64)
    frame1 = 255.0 * (1.0 - WINDOW_DIFFUSE_DEMO_ALPHA_D * mask) + rng.normal(0.0, DIFFUSE_SWEEP_NOISE_STD, SWEEP_SHAPE)
    frame0_u8 = frame0.astype(np.uint8)
    frame1_u8 = np.clip(frame1, 0.0, 255.0).astype(np.uint8)

    owner = make_standalone_owner(SWEEP_AVG_CRACK_WIDTH_PX)
    variants: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for window in WINDOW_DIFFUSE_VARIANTS:
        params = dict(DETECTION_DIFFUSE_PARAMS)
        params["window_diffuse"] = window
        result = DiffuseDetector(owner).diffuse_delamination(
            cracks=[np.zeros((0, 2, 2)), SWEEP_CRACK],
            processed_stack=[frame0_u8, frame1_u8],
            params=params,
            save_overlays=False,
        )
        detected = np.asarray(result["masks"]["frame_0001"], dtype=bool)
        variants[window] = {"mask": detected, "iou": iou_score(detected, mask)}

    return {"frame": frame1_u8, "gt_mask": mask, "variants": variants}


def make_window_diffuse_figure(demo: Dict[str, Any], *, output_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    _configure_style()

    fig, axes = plt.subplots(1, 4, figsize=(10.5, 4.6))
    for ax, window in zip(axes, WINDOW_DIFFUSE_VARIANTS):
        info = demo["variants"][window]
        ax.imshow(demo["frame"], cmap="gray", vmin=0, vmax=255)
        overlay = np.zeros((*info["mask"].shape, 4), dtype=np.float32)
        overlay[info["mask"], 1] = 0.7
        overlay[info["mask"], 3] = 0.55
        ax.imshow(overlay)
        ax.contour(demo["gt_mask"], levels=[0.5], colors="black", linewidths=0.8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"window_diffuse={window}",
            fontsize=8 * WINDOW_FIGURE_FONT_SCALE,
        )
    fig.suptitle(
        "Effect of window_diffuse at fixed contrast\n"
        r"($\alpha_d$="
        f"{WINDOW_DIFFUSE_DEMO_ALPHA_D}, width scale={WINDOW_DIFFUSE_DEMO_WIDTH_SCALE:g}"
        r"$\times$avg_crack_width_px)",
        fontsize=10 * WINDOW_FIGURE_FONT_SCALE,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.76))

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "synthetic_validation_window_diffuse.png"
    pdf_path = output_dir / "synthetic_validation_window_diffuse.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path


def run_default_vs_scale_matched_on_main_sequence(*, results_root: Path) -> Dict[str, float]:
    """Compare window_diffuse=(0, 60) [package default] vs (1, 20) [this example]
    on the *real* main-example geometry (all 6 cracks, real rendered noise,
    real static-reference preprocessing) rather than the small synthetic
    sweep case above.
    """
    from deladect.specimen import Specimen
    from synthetic_delamination_validation import _write_frames_to_disk

    sequence = generate_synthetic_sequence()
    frames_dir = results_root / "main_sequence_frames" / "full"
    _write_frames_to_disk(sequence["frames"], frames_dir)

    specimen = Specimen(
        name="synthetic-validation-window-check",
        scale_px_mm=sequence["params"]["scale_px_mm"],
        path_full=str(frames_dir),
        sorting_key="_sc",
        image_types=["png"],
        results_root=str(results_root),
        avg_crack_width_px=8.0,
    )
    specimen.add_ply(name="ply_0", orientation_deg=0.0, avg_crack_width_px=8.0, min_crack_length_px=20.0)
    specimen.add_ply(name="ply_1", orientation_deg=90.0, avg_crack_width_px=8.0, min_crack_length_px=20.0)
    specimen.add_interface(name="synthetic", upper_ply=0, lower_ply=1)

    from deladect.detection.delamination.core import DelaminationDetector

    detector = DelaminationDetector(specimen, specimen.interfaces[0])
    cache_paths = detector.preprocess_stack_to_disk(
        specimen.image_stack_full, key="window_diffuse_check", reference_mode="static",
    )["cache_paths"]

    gt = sequence["diffuse_raw_gt_cum"][-1]
    scores: Dict[str, float] = {}
    for label, window in {"package_default": (0, 60), "scale_matched": (1, 20)}.items():
        params = dict(DETECTION_DIFFUSE_PARAMS)
        params["window_diffuse"] = window
        result = detector.diffuse.diffuse_delamination(
            cracks=sequence["cracks_by_frame"], processed_cache_paths=cache_paths,
            params=params, save_overlays=False,
        )
        last_key = sorted(result["masks"].keys())[-1]
        detected = np.asarray(result["masks"][last_key], dtype=bool)
        scores[label] = iou_score(detected, gt)
    return scores


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    DOCS_STATIC_DIR.mkdir(parents=True, exist_ok=True)

    print("Running diffuse detectability sweep "
          f"({len(DIFFUSE_CONTRAST_VALUES)}x{len(DIFFUSE_WIDTH_SCALES)}x{len(DIFFUSE_SWEEP_SEEDS)} cells)...")
    diffuse_df = run_diffuse_detectability_sweep()
    diffuse_df.to_csv(RESULTS_ROOT / "diffuse_detectability_sweep.csv", index=False)

    print("Running edge detectability sweep "
          f"({len(EDGE_CONTRAST_VALUES)}x{len(EDGE_NOISE_STD_VALUES)}x{len(EDGE_SWEEP_SEEDS)} cells)...")
    edge_df = run_edge_detectability_sweep()
    edge_df.to_csv(RESULTS_ROOT / "edge_detectability_sweep.csv", index=False)

    detectability_png = make_detectability_figure(diffuse_df, edge_df, output_dir=RESULTS_ROOT)
    print(f"Detectability figure: {detectability_png}")

    example_png = make_sweep_example_figure(output_dir=RESULTS_ROOT)
    print(f"Sweep example figure: {example_png}")

    print("Running window_diffuse parameter-scale demo...")
    demo = run_window_diffuse_demo()
    window_png = make_window_diffuse_figure(demo, output_dir=RESULTS_ROOT)
    print(f"window_diffuse figure: {window_png}")
    print("window_diffuse demo IoU by setting:")
    for window in WINDOW_DIFFUSE_VARIANTS:
        print(f"  {window}: IoU={demo['variants'][window]['iou']:.3f}")

    print("Comparing package-default vs. scale-matched window_diffuse on the main sequence...")
    comparison = run_default_vs_scale_matched_on_main_sequence(results_root=RESULTS_ROOT)
    print(f"  package_default (0, 60): diffuse IoU = {comparison['package_default']:.3f}")
    print(f"  scale_matched   (1, 20): diffuse IoU = {comparison['scale_matched']:.3f}")
    pd.Series(comparison).to_csv(RESULTS_ROOT / "window_diffuse_main_sequence_comparison.csv")

    for suffix in (".png", ".pdf"):
        for stem in (
            "synthetic_validation_detectability",
            "synthetic_validation_sweep_examples",
            "synthetic_validation_window_diffuse",
        ):
            src = RESULTS_ROOT / f"{stem}{suffix}"
            if src.exists():
                import shutil
                shutil.copy2(src, DOCS_STATIC_DIR / f"{stem}{suffix}")
    for stem in ("diffuse_detectability_sweep", "edge_detectability_sweep", "window_diffuse_main_sequence_comparison"):
        src = RESULTS_ROOT / f"{stem}.csv"
        if src.exists():
            import shutil
            shutil.copy2(src, DOCS_STATIC_DIR / f"{stem}.csv")
    print(f"Documentation copies written to: {DOCS_STATIC_DIR}")


if __name__ == "__main__":
    main()
