"""Evaluate a specimen with explicit upper, middle, and lower image regions."""

from pathlib import Path

from deladect.detection import DelaminationDetector, crack_analysis
from deladect.io import save_specimen
from deladect.specimen import Specimen


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "example_images" / "sample-2"
RESULTS_ROOT = REPO_ROOT / "results"


def main() -> None:
    specimen = Specimen.from_cross_ply(
        name="02-advanced-options",
        scale_px_mm=41.03328366,
        path_full=str(DATA_ROOT / "full"),
        path_upper_border=str(DATA_ROOT / "upper"),
        path_middle=str(DATA_ROOT / "middle"),
        path_lower_border=str(DATA_ROOT / "lower"),
        sorting_key="_sc",
        image_types=["png"],
        results_root=str(RESULTS_ROOT),
        avg_crack_width_px=8.0,
        strain_csv=str(DATA_ROOT / "experimental_data.csv"),
    )

    # Auto-selection uses the middle region for cracks.
    crack_results = crack_analysis(
        specimen,
        export_images=True,
        background=True,
        save_cracks=True,
    )
    # Edge detection uses only upper/lower; output masks are reassembled at full size.
    detector = DelaminationDetector(
        specimen,
        specimen.interfaces[0],
        save_preprocess_outputs=True,
    )
    result = detector.detect_both_delaminations(
        cracks=crack_results,
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
