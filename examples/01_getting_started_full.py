"""Run the Getting Started analysis using only the full-frame image stack."""

from pathlib import Path

from deladect.detection import DelaminationDetector, crack_analysis
from deladect.io import save_specimen
from deladect.specimen import Specimen


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results"


def main() -> None:
    data_root = REPO_ROOT / "example_images" / "sample-1"
    specimen = Specimen(
        name="01-getting-started-full",
        scale_px_mm=31.953,
        path_full=str(data_root / "full"),
        sorting_key="_sc",
        image_types=["png"],
        results_root=str(RESULTS_ROOT),
        avg_crack_width_px=8.0,
    )
    ply0 = specimen.add_ply(
        name="ply_0",
        orientation_deg=0.0,
        avg_crack_width_px=8.0,
        min_crack_length_px=90.0,
    )
    ply90 = specimen.add_ply(
        name="ply_90",
        orientation_deg=90.0,
        avg_crack_width_px=8.0,
        min_crack_length_px=90.0,
    )
    interface = specimen.add_interface(
        name="i0",
        upper_ply=ply0,
        lower_ply=ply90,
    )

    crack_results = crack_analysis(
        specimen,
        export_images=True,
        background=True,
        save_cracks=True,
    )
    detector = DelaminationDetector(
        specimen,
        interface,
        save_preprocess_outputs=True,
    )
    result = detector.detect_both_delaminations(
        cracks=crack_results,
        save_overlays=True,
        overlay_view="classified",
        save_component_overlays=True,
        save_masks=True,
        save_metrics=True,
        diffuse_params={
            "window_diffuse": (30, 30),
            "diffuse_dx": 40.0,
            "diffuse_dy": 10.0,
        },
        edge_params={
            "window_edge": (1, 90),
            "seed_ratio": 0.01,
        },
        progress=True,
    )

    manifest = specimen.results_dir("config") / "specimen.json"
    save_specimen(specimen, manifest)
    print(f"Results: {specimen.results_root_path()}")
    print(f"Combined overlays: {result['paths']['combined_overlays']}")


if __name__ == "__main__":
    main()
