"""Compare static-reference and rolling-median-reference preprocessing on Sample-3.

Requires examples/02_multi_interface_edge_delamination.py to have been run
first, since it populates the Preprocessor_cache this script reads from.

Uses the same plot_style template (SciencePlots + LaTeX + A4-based sizing)
as results/generate_multi_interface_documentation_outputs.py, so this figure
matches the rest of the page.
"""

from pathlib import Path
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


ROOT = Path(__file__).resolve().parents[1]
MY_PLOTS_ROOT = ROOT.parent / "my_plots"
for source_root in (ROOT / "src", MY_PLOTS_ROOT):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from plot_style import get_figsize, save_fig, setup  # noqa: E402

RESULTS = ROOT / "results" / "02-multi-interface-edge"
CACHE = RESULTS / "Preprocessor_cache"
OUTPUT_DIR = RESULTS / "documentation"
DOCS_OUTPUT_DIR = ROOT / "docs" / "source" / "_static" / "examples"

FRAME_NUMBERS = np.array([2, 80, 195, 206, 217, 228, 239, 250, 261, 272], dtype=int)
DISPLAY_INDEX = len(FRAME_NUMBERS) - 1
FRAME_NUMBER = int(FRAME_NUMBERS[DISPLAY_INDEX])


def load_frame(key: str) -> dict:
    path = CACHE / key / f"preprocess_{DISPLAY_INDEX:04d}.npz"
    with np.load(path, allow_pickle=False) as payload:
        return {"baseline": payload["baseline"], "processed": payload["processed"]}


def configure_image_axis(axis: plt.Axes, title: str, shape: tuple[int, int]) -> None:
    height, width = shape
    axis.set_title(title, fontsize=10, pad=4)
    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)
    axis.set_aspect("equal")
    axis.set_xlabel(r"$x$ [px]", fontsize=9)
    axis.set_ylabel(r"$y$ [px]", fontsize=9)
    axis.xaxis.set_major_locator(MultipleLocator(500))
    axis.yaxis.set_major_locator(MultipleLocator(400))
    axis.tick_params(axis="both", which="major", labelsize=8, length=2.5)
    axis.minorticks_off()


def add_row_label(figure: plt.Figure, grid, row: int, text: str) -> None:
    axis = figure.add_subplot(grid[row, :])
    axis.axis("off")
    axis.text(0.5, 0.0, text, ha="center", va="bottom", fontsize=11, transform=axis.transAxes)


def main() -> Path:
    setup(mode="light", usetex=True, two_col=False)

    static = load_frame("primary_static")
    rolling = load_frame("secondary_rolling")
    shape = static["baseline"].shape

    figure = plt.figure(figsize=get_figsize(two_col=False, aspect=0.68))
    grid = figure.add_gridspec(4, 2, height_ratios=(0.16, 1.0, 0.16, 1.0))

    add_row_label(figure, grid, 0, "(a) Static reference")

    ax_static_baseline = figure.add_subplot(grid[1, 0])
    ax_static_normalized = figure.add_subplot(grid[1, 1])
    ax_static_baseline.imshow(static["baseline"], cmap="gray", vmin=0, vmax=255)
    configure_image_axis(ax_static_baseline, "Baseline", shape)
    ax_static_normalized.imshow(static["processed"], cmap="gray", vmin=0, vmax=255)
    configure_image_axis(ax_static_normalized, "Normalized", shape)

    add_row_label(figure, grid, 2, "(b) Rolling-median reference")

    ax_rolling_baseline = figure.add_subplot(grid[3, 0])
    ax_rolling_normalized = figure.add_subplot(grid[3, 1])
    ax_rolling_baseline.imshow(rolling["baseline"], cmap="gray", vmin=0, vmax=255)
    configure_image_axis(ax_rolling_baseline, "Baseline", shape)
    ax_rolling_normalized.imshow(rolling["processed"], cmap="gray", vmin=0, vmax=255)
    configure_image_axis(ax_rolling_normalized, "Normalized", shape)

    stem = "static_vs_rolling_median_preprocessing"
    save_fig(figure, stem, path=OUTPUT_DIR, formats=["pdf", "svg", "png"], transparent=True)

    output_path = OUTPUT_DIR / f"{stem}.png"
    DOCS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output_path, DOCS_OUTPUT_DIR / output_path.name)
    print(f"Frame: {FRAME_NUMBER} (sample position {DISPLAY_INDEX})")
    print(f"Result figure: {output_path}")
    print(f"Documentation figure: {DOCS_OUTPUT_DIR / output_path.name}")
    return output_path


if __name__ == "__main__":
    main()
