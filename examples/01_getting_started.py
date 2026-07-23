"""Run the smallest complete crack and delamination analysis."""

from pathlib import Path

from deladect.detection import DelaminationDetector, crack_analysis
from deladect.io import save_specimen
from deladect.specimen import Specimen


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results"


def main() -> None:
    data_root = REPO_ROOT / "example_images" / "sample-1"
    specimen = Specimen(
        name="01-getting-started",
        scale_px_mm=41.03328366,
        path_full=str(data_root / "full"),
        path_upper_border=str(data_root / "upper"),
        path_middle=str(data_root / "middle"),
        path_lower_border=str(data_root / "lower"),
        sorting_key="_sc",
        image_types=["png"],
        results_root=str(RESULTS_ROOT),
        avg_crack_width_px=8.0,
    )
    specimen.add_ply(name="ply_0", orientation_deg=0.0)
    specimen.add_ply(name="ply_90", orientation_deg=90.0)
    interface = specimen.add_interface(name="i0", upper_ply_index=0, lower_ply_index=1)

    crack_results = crack_analysis(
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
        progress=True,
    )

    manifest = specimen.results_dir("config") / "specimen.json"
    save_specimen(specimen, manifest)
    print(f"Results: {specimen.results_root_path()}")
    print(f"Combined overlays: {result['paths']['combined_overlays']}")


if __name__ == "__main__":
    main()
