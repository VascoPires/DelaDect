"""Run the smallest complete crack and delamination analysis."""

from pathlib import Path

from deladect.detection import DelaminationDetector, crack_eval_crossply
from deladect.io import save_specimen
from deladect.specimen import Specimen


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results"


def main() -> None:
    specimen = Specimen.from_cross_ply(
        name="01-getting-started",
        scale_px_mm=41.03328366,
        path_full=str(REPO_ROOT / "example_images" / "sample-1"),
        sorting_key="_sc",
        image_types=["png"],
        results_root=str(RESULTS_ROOT),
        avg_crack_width_px=8.0,
    )
    interface = specimen.interfaces[0]

    crack_results = crack_eval_crossply(
        specimen,
        export_images=True,
        background=True,
        save_cracks=True,
    )
    cracks = Specimen.join_cracks(
        crack_results["0"]["cracks"],
        crack_results["90"]["cracks"],
    )

    detector = DelaminationDetector(
        specimen,
        interface,
        save_preprocess_outputs=True,
    )
    result = detector.detect_both_delaminations(
        cracks=cracks,
        avg_crack_width_px=8.0,
        save_overlays=True,
        overlay_view="classified",
        save_component_overlays=True,
        save_masks=True,
        save_metrics=True,
        edge_exclusion_px=5,
        diffuse_params={"window_diffuse": (60, 60)},
        progress=True,
    )

    manifest = specimen.results_dir("config") / "specimen.json"
    save_specimen(specimen, manifest)
    print(f"Results: {specimen.results_root_path()}")
    print(f"Combined overlays: {result['paths']['combined_overlays']}")


if __name__ == "__main__":
    main()
