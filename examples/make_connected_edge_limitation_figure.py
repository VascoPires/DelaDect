"""Create a static square-cell diagram of edge precedence in Sample-1."""

from pathlib import Path
import shutil

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy import ndimage as ndi


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "01-getting-started"
MASKS = RESULTS / "delamination" / "both" / "masks"
OUTPUT = RESULTS / "limitations" / "connected_edge_square_masks.svg"
DOCS_OUTPUT = ROOT / "docs" / "source" / "_static" / "examples" / "connected_edge_square_masks.svg"
FRAME_KEY = "frame_0003"

# Full specimen height and a 600-pixel-wide region containing clear vertical
# diffuse stripes. Padding one row gives exact 30x30 source-pixel cells.
COLUMN_SLICE = slice(1200, 1800)
CELL_SIZE = 30

WHITE = 0
EDGE_LIGHT = 1
EDGE_RED = 2
DIFFUSE_GREEN = 3
CELL_COLORS = {
    WHITE: "#FFFFFF",
    EDGE_LIGHT: "#F6C8C8",
    EDGE_RED: "#D73031",
    DIFFUSE_GREEN: "#15965A",
}


def load_mask(name: str) -> np.ndarray:
    with np.load(MASKS / f"{name}.npz") as payload:
        return np.asarray(payload[FRAME_KEY], dtype=bool)


def block_fraction(mask: np.ndarray) -> np.ndarray:
    cropped = mask[:, COLUMN_SLICE]
    padded = np.pad(cropped, ((0, 2), (0, 0)), constant_values=False)
    height, width = padded.shape
    return padded.reshape(
        height // CELL_SIZE,
        CELL_SIZE,
        width // CELL_SIZE,
        CELL_SIZE,
    ).mean(axis=(1, 3))


def draw_grid(axis: plt.Axes, cells: np.ndarray, title: str) -> None:
    rows, columns = cells.shape
    axis.set_facecolor(CELL_COLORS[WHITE])
    for row in range(rows):
        for column in range(columns):
            value = int(cells[row, column])
            if value == WHITE:
                continue
            axis.add_patch(
                Rectangle(
                    (column - 0.5, row - 0.5),
                    1.0,
                    1.0,
                    facecolor=CELL_COLORS[value],
                    edgecolor="none",
                )
            )
    for boundary in np.arange(-0.5, columns + 0.5, 1.0):
        axis.axvline(boundary, color="#FFFFFF", linewidth=1.15)
    for boundary in np.arange(-0.5, rows + 0.5, 1.0):
        axis.axhline(boundary, color="#FFFFFF", linewidth=1.15)
    axis.set_xlim(-0.5, columns - 0.5)
    axis.set_ylim(rows - 0.5, -0.5)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(title, fontsize=13, pad=10)
    for spine in axis.spines.values():
        spine.set_color("#555555")
        spine.set_linewidth(1.0)


def main() -> None:
    edge = load_mask("edge_exclusion")
    diffuse_raw = load_mask("diffuse_raw")
    diffuse_final = load_mask("diffuse_final")

    labels, _ = ndi.label(edge)
    boundary_labels = (set(np.unique(labels[0])) & set(np.unique(labels[-1]))) - {0}
    if not boundary_labels:
        raise RuntimeError("No edge component connects the upper and lower boundaries.")
    connected_label = max(boundary_labels, key=lambda label: np.count_nonzero(labels == label))
    connected = labels == connected_label

    edge_fraction = block_fraction(edge)
    connected_fraction = block_fraction(connected)
    diffuse_fraction = block_fraction(diffuse_raw)
    final_fraction = block_fraction(diffuse_final)

    # A cell represents 30x30 source pixels. Low thresholds retain narrow
    # vertical stripes while the color categories remain exact mask classes.
    edge_cells = edge_fraction >= 0.08
    connected_cells = connected_fraction >= 0.08
    diffuse_cells = diffuse_fraction > 0.0
    final_cells = final_fraction > 0.0

    panel_edge = np.where(connected_cells, EDGE_RED, WHITE)
    panel_candidates = np.where(edge_cells, EDGE_LIGHT, WHITE)
    panel_candidates[diffuse_cells] = DIFFUSE_GREEN
    panel_final = np.where(edge_cells | diffuse_cells, EDGE_RED, WHITE)
    panel_final[final_cells] = DIFFUSE_GREEN

    figure, axes = plt.subplots(1, 3, figsize=(11.5, 7.2))
    figure.subplots_adjust(left=0.12, right=0.98, top=0.85, bottom=0.18, wspace=0.08)
    draw_grid(axes[0], panel_edge, "1. Connected edge component")
    draw_grid(axes[1], panel_candidates, "2. Before precedence")
    draw_grid(axes[2], panel_final, "3. Final classification")

    axes[0].annotate(
        "lower edge",
        xy=(-0.05, 0.02),
        xytext=(-0.32, 0.02),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "#555555"},
        ha="right",
        va="center",
        fontsize=10,
    )
    axes[0].annotate(
        "upper edge",
        xy=(-0.05, 0.98),
        xytext=(-0.32, 0.98),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "#555555"},
        ha="right",
        va="center",
        fontsize=10,
    )

    overlap = edge & diffuse_raw
    overlap_fraction = float(overlap.sum() / max(1, diffuse_raw.sum()))
    figure.suptitle(
        "Connected edge delamination overwrites diffuse classification",
        fontsize=18,
        y=0.95,
    )
    figure.text(
        0.5,
        0.09,
        "Red: edge   |   Pale red: edge shown beneath candidates   |   "
        "Green: diffuse detected before precedence",
        ha="center",
        fontsize=11,
    )
    figure.text(
        0.5,
        0.045,
        f"Frame 0003: {overlap_fraction:.2%} of diffuse-candidate pixels overlap edge and are assigned to edge.",
        ha="center",
        fontsize=11,
        weight="bold",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DOCS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(figure)
    shutil.copyfile(OUTPUT, DOCS_OUTPUT)
    print(f"Result figure: {OUTPUT}")
    print(f"Documentation figure: {DOCS_OUTPUT}")


if __name__ == "__main__":
    main()
