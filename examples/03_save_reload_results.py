"""Reload a saved specimen: manifest, stored results, and experimental data."""

from pathlib import Path

from deladect.io import load_specimen, load_stored_results


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "results" / "01-getting-started" / "config" / "specimen.json"


def main() -> None:
    if not MANIFEST.exists():
        raise FileNotFoundError(
            "Run examples/01_getting_started.py before this persistence example."
        )

    specimen = load_specimen(
        MANIFEST,
        auto_init_stacks=False,
        load_results=True,
        verbose=True,
        strict=True,
    )
    bundles = load_stored_results(specimen, strict=True, verbose=True)
    print(f"Reloaded specimen: {specimen.name}")
    print(f"Stored result groups: {sorted(bundles)}")

    # strain_csv is part of the saved manifest, so the strain data reloads
    # automatically -- no separate step is needed to bring it back.
    if specimen.experimental_data is not None:
        print("Reloaded experimental data (strain_y):")
        print(specimen.experimental_data.to_string(index=False))
    else:
        print("No experimental data was attached to this specimen.")


if __name__ == "__main__":
    main()
