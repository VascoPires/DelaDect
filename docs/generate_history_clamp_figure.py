"""Generate the synthetic history-clamp figure used by the documentation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from deladect.detection.delamination._preprocess import PreprocessingMixin


OUTPUT = Path(__file__).parent / "source" / "_static" / "normalization" / "history_clamp_noise.png"
N_FRAMES = 25
SHAPE = (96, 96)
SAMPLE_FRAMES = (0, 4, 9, 16, 24)
BACKGROUND_LEVEL = 0.68
NOISE_SIGMA = 0.075
SEED = 17


class _HistoryProcessor(PreprocessingMixin):
    """Minimal owner used to call the production history-clamp method."""

    def __init__(self) -> None:
        self._notice_flags: dict[str, bool] = {}


def _synthetic_stack() -> tuple[list[np.ndarray], np.ndarray]:
    """Return noisy frames and a mask selecting unchanged background pixels."""
    rng = np.random.default_rng(SEED)
    yy, xx = np.ogrid[: SHAPE[0], : SHAPE[1]]
    blobs = ((34, 30, 13), (30, 64, 10), (66, 55, 15))
    damage = np.zeros(SHAPE, dtype=bool)
    for cy, cx, radius in blobs:
        damage |= (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2

    stack: list[np.ndarray] = []
    for _ in range(N_FRAMES):
        frame = BACKGROUND_LEVEL + rng.normal(0.0, NOISE_SIGMA, SHAPE)
        frame = np.clip(frame, 0.0, 1.0)
        frame[damage] = 0.035
        stack.append(np.round(frame * 255.0).astype(np.uint8))
    return stack, ~damage


def _background_summary(
    stack: list[np.ndarray], background: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return median, 10th percentile, and 90th percentile by frame."""
    values = np.asarray([frame[background] for frame in stack], dtype=float) / 255.0
    return (
        np.median(values, axis=1),
        np.percentile(values, 10, axis=1),
        np.percentile(values, 90, axis=1),
    )


def main() -> None:
    raw_stack, background = _synthetic_stack()
    clamped_stack = _HistoryProcessor().apply_minimum_history(
        raw_stack,
        key="documentation_history_clamp",
        history_buffers={},
        mode="running",
    )["frames"]

    raw_median, raw_p10, raw_p90 = _background_summary(raw_stack, background)
    clamp_median, clamp_p10, clamp_p90 = _background_summary(clamped_stack, background)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 16,
            "axes.titlesize": 18,
            "axes.labelsize": 18,
            "axes.labelweight": "normal",
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 16,
        }
    )
    fig = plt.figure(figsize=(15.5, 14.2))
    fig.subplots_adjust(
        left=0.065,
        right=0.985,
        bottom=0.035,
        top=0.985,
        wspace=0.16,
        hspace=0.10,
    )
    grid = fig.add_gridspec(
        5,
        10,
        height_ratios=(1.0, 1.0, 0.28, 2.25, 1.05),
    )
    plot_grid = grid[3, :].subgridspec(1, 2, wspace=0.18)
    legend_grid = grid[4, :].subgridspec(1, 2, wspace=0.04)

    for column, frame_idx in enumerate(SAMPLE_FRAMES):
        raw_ax = fig.add_subplot(grid[0, 2 * column : 2 * column + 2])
        clamp_ax = fig.add_subplot(grid[1, 2 * column : 2 * column + 2])
        for axis, frame in (
            (raw_ax, raw_stack[frame_idx]),
            (clamp_ax, clamped_stack[frame_idx]),
        ):
            axis.imshow(frame, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_linewidth(1.4)
        if column == 0:
            raw_ax.set_ylabel("Unclamped", fontsize=19)
            clamp_ax.set_ylabel("History clamp", fontsize=19)

    hist_ax = fig.add_subplot(plot_grid[0, 0])
    bins = np.linspace(0.34, 1.0, 45)
    blues = plt.cm.Blues(np.linspace(0.35, 0.90, 4))
    for color, frame_idx in zip(blues, (0, 4, 9, 24)):
        hist_ax.hist(
            clamped_stack[frame_idx][background] / 255.0,
            bins=bins,
            histtype="step",
            linewidth=2.2,
            color=color,
            label=f"clamped: frame {frame_idx}",
        )
    hist_ax.hist(
        raw_stack[0][background] / 255.0,
        bins=bins,
        histtype="step",
        linewidth=2.2,
        linestyle="--",
        color="#D62728",
        label="unclamped: frame 0",
    )
    hist_ax.set_xlabel("Pixel greyscale value")
    hist_ax.set_ylabel("Pixel count")
    hist_ax.set_xlim(bins[0], bins[-1])
    hist_handles, hist_labels = hist_ax.get_legend_handles_labels()

    summary_ax = fig.add_subplot(plot_grid[0, 1])
    frames = np.arange(N_FRAMES)
    summary_ax.fill_between(
        frames,
        raw_p10,
        raw_p90,
        color="#D62728",
        alpha=0.16,
        linewidth=0,
        label="unclamped: 10th–90th percentile",
    )
    summary_ax.plot(
        frames,
        raw_median,
        color="#D62728",
        linestyle="--",
        linewidth=2.2,
        label="unclamped: median",
    )
    summary_ax.fill_between(
        frames,
        clamp_p10,
        clamp_p90,
        color="#1F77B4",
        alpha=0.22,
        linewidth=0,
        label="history clamp: 10th–90th percentile",
    )
    summary_ax.plot(
        frames,
        clamp_median,
        color="#1F77B4",
        linewidth=2.4,
        label="history clamp: median",
    )
    summary_ax.set_xlabel("Frame index")
    summary_ax.set_ylabel("Background greyscale value")
    summary_ax.set_xlim(0, N_FRAMES - 1)
    summary_ax.set_ylim(0.42, 0.82)
    summary_handles, summary_labels = summary_ax.get_legend_handles_labels()

    hist_legend_ax = fig.add_subplot(legend_grid[0, 0])
    hist_legend_ax.axis("off")
    hist_legend_ax.legend(
        hist_handles,
        hist_labels,
        frameon=False,
        loc="center",
        ncol=2,
    )
    summary_legend_ax = fig.add_subplot(legend_grid[0, 1])
    summary_legend_ax.axis("off")
    summary_legend_ax.legend(
        summary_handles,
        summary_labels,
        frameon=False,
        loc="center",
        ncol=1,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, facecolor="white")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
