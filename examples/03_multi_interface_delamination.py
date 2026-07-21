"""Run hierarchical edge delamination over three ordered interfaces."""

from pathlib import Path

from deladect.detection import DelaminationDetector
from deladect.io import save_specimen
from deladect.specimen import Specimen


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results"


def main() -> None:
    specimen = Specimen(
        name="03-multi-interface",
        scale_px_mm=41.03328366,
        path_full=str(REPO_ROOT / "example_images" / "sample-4"),
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
            upper_ply_index=index,
            lower_ply_index=index + 1,
        )

    detector = DelaminationDetector(
        specimen,
        specimen.interfaces[0],
        save_preprocess_outputs=True,
    )
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
    result = detector.edge.detect_edge_multi(
        interfaces=specimen.interfaces,
        processed_cache_paths=primary_cache,
        secondary_cache_paths=secondary_cache,
        save_masks=True,
        save_overlays=True,
        secondary_params={
            "secondary_similarity_threshold": 0.6,
        },
    )

    manifest = specimen.results_dir("config") / "specimen.json"
    save_specimen(specimen, manifest)
    print(f"Results: {specimen.results_root_path()}")
    print(f"Multi-interface overlays: {result['paths']['overlays']}")


if __name__ == "__main__":
    main()
