"""Compare static-reference and rolling-median-reference preprocessing on Sample-3.

Requires examples/02_multi_interface_edge_delamination.py to have been run
first, since it populates the Preprocessor_cache this script reads from.
"""

from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "02-multi-interface-edge"
CACHE = RESULTS / "Preprocessor_cache"
OUTPUT = RESULTS / "static_vs_rolling_median_preprocessing.png"
DOCS_OUTPUT = ROOT / "docs" / "source" / "_static" / "examples" / "static_vs_rolling_median_preprocessing.png"

FRAME_IDX = 9  # last sampled frame, where the two references have diverged the most


def load_frame(key: str, frame_idx: int) -> dict:
    path = CACHE / key / f"preprocess_{frame_idx:04d}.npz"
    with np.load(path, allow_pickle=False) as payload:
        return {"baseline": payload["baseline"], "processed": payload["processed"]}


def main() -> None:
    static = load_frame("primary_static", FRAME_IDX)
    rolling = load_frame("secondary_rolling", FRAME_IDX)

    fig, axes = plt.subplots(2, 2, figsize=(14, 6.4), constrained_layout=True)

    axes[0, 0].imshow(static["baseline"], cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Static reference: baseline")
    axes[0, 1].imshow(static["processed"], cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("Static reference: processed")

    axes[1, 0].imshow(rolling["baseline"], cmap="gray", vmin=0, vmax=255)
    axes[1, 0].set_title("Rolling-median reference: baseline")
    axes[1, 1].imshow(rolling["processed"], cmap="gray", vmin=0, vmax=255)
    axes[1, 1].set_title("Rolling-median reference: processed")

    for ax in axes.flat:
        ax.axis("off")

    fig.suptitle(
        f"Sample-3, frame {FRAME_IDX}: static vs. rolling-median (window=7, skip=2) preprocessing",
        fontsize=13,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DOCS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=150, facecolor="white")
    plt.close(fig)
    shutil.copyfile(OUTPUT, DOCS_OUTPUT)
    print(f"Result figure: {OUTPUT}")
    print(f"Documentation figure: {DOCS_OUTPUT}")


if __name__ == "__main__":
    main()
