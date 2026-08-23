"""Edge-only delamination: single interface, then multi-interface promotion."""

from pathlib import Path

from deladect.detection import DelaminationDetector
from deladect.io import save_specimen
from deladect.io.delamination import save_mask_bundle
from deladect.specimen import Specimen


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results"


def main() -> None:
    specimen = Specimen(
        name="02-multi-interface-edge",
        scale_px_mm=41.03328366,
        path_full=str(REPO_ROOT / "example_images" / "sample-3"),
        sorting_key="_sc",
        image_types=["png"],
        results_root=str(RESULTS_ROOT),
        avg_crack_width_px=8.0,
    )
    for index, orientation in enumerate((0.0, 90.0, 0.0)):
        specimen.add_ply(
            name=f"ply_{index}",
            orientation_deg=orientation,
            avg_crack_width_px=8.0,
            min_crack_length_px=20.0,
        )
    for index in range(2):
        specimen.add_interface(
            name=f"i{index}",
            upper_ply=index,
            lower_ply=index + 1,
        )

    detector = DelaminationDetector(
        specimen,
        specimen.interfaces[0],
        save_preprocess_outputs=True,
    )

    # 1. Standalone edge delamination on a single interface (i0). No crack
    #    catalogue and no diffuse pipeline are involved -- detect_primary()
    #    can run entirely on its own.
    primary_only = detector.edge.detect_primary(
        save_overlays=True,
        overlay_dirname="edge_only",
        params={
            "window_edge": (1, 130),
            "gaussian_filters": (0.5, 15.0),
            "hard_floor": 0.90,
            "scale_min_percentile": 10,
            "scale_max_percentile": 95,
            "seed_ratio": 0.01,
            "post_threshold_closing_px": 20,
        },
    )
    primary_only_masks_path = save_mask_bundle(
        primary_only["masks"],
        specimen.results_dir("edge_only", "edge", "masks") / "primary.npz",
    )
    primary_only_overlays = specimen.results_dir("edge_only", "edge", "overlays")
    print(f"Single-interface (i0) edge masks: {primary_only_masks_path}")
    print(f"Single-interface (i0) edge overlays: {primary_only_overlays}")

    # 2. Multi-interface promotion across i0 (primary) and i1 (secondary).
    #    The primary accumulation uses a static reference; the secondary
    #    promotion pass uses a rolling-median reference so it stays sensitive
    #    to change inside the already-established primary region.
    primary_cache = detector.preprocess_stack_to_disk(
        specimen.image_stack_full,
        key="primary_static",
        reference_mode="static",
    )["cache_paths"]
    secondary_cache = detector.preprocess_stack_to_disk(
        specimen.image_stack_full,
        key="secondary_rolling",
        reference_mode="rolling_median",
        reference_window=7,
        reference_skip=2,
    )["cache_paths"]
    multi_result = detector.edge.detect_edge_multi(
        interfaces=specimen.interfaces,
        processed_cache_paths=primary_cache,
        secondary_cache_paths=secondary_cache,
        save_masks=True,
        save_overlays=True,
        primary_params={
            "window_edge": (1, 130),
            "gaussian_filters": (0.5, 15.0),
            "hard_floor": 0.90,
            "scale_min_percentile": 10,
            "scale_max_percentile": 95,
            "seed_ratio": 0.01,
            "post_threshold_closing_px": 20,
        },
        secondary_edge_params={
            "window_edge": (1, 30),
            "gaussian_filters": (0.5, 15.0),
            "hard_floor": 0.90,
            "scale_min_percentile": 10,
            "scale_max_percentile": 95,
            "seed_ratio": 0.01,
            "post_threshold_closing_px": 10,
        },
        secondary_params={
            "secondary_similarity_threshold": 0.80,
            "min_primary_frac_for_secondary": 0.10,
            # Position 2 in the sampled stack == frame 195, the closest sample-3
            # frame to Out30-p1's secondary_start_frame_id=181 in its own numbering.
            "secondary_start_frame": 2,
        },
    )

    manifest = specimen.results_dir("config") / "specimen.json"
    save_specimen(specimen, manifest)
    print(f"Results: {specimen.results_root_path()}")
    print(f"Multi-interface overlays: {multi_result['paths']['overlays']}")


if __name__ == "__main__":
    main()
