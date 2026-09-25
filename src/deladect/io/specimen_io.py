"""Save a :class:`~deladect.specimen.Specimen` to JSON and load it back."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from .cracks import PLY_CRACK_RESULTS_KEY, load_ply_crack_results
from .delamination import (
    INTERFACE_COMBINED_MASKS_KEY,
    INTERFACE_DIFFUSE_MASKS_KEY,
    INTERFACE_DIFFUSE_RAW_MASKS_KEY,
    INTERFACE_METRICS_KEY,
    INTERFACE_PRIMARY_MASKS_KEY,
    INTERFACE_SECONDARY_MASKS_KEY,
    load_interface_combined_masks,
    load_interface_diffuse_masks,
    load_interface_diffuse_raw_masks,
    load_interface_metrics,
    load_interface_primary_masks,
    load_interface_secondary_masks,
)

from deladect.specimen import Specimen

JsonLikePath = Union[str, Path]

# (summary label, metadata key, loader, report key)
_INTERFACE_ARTEFACTS = (
    ("edge", INTERFACE_PRIMARY_MASKS_KEY, load_interface_primary_masks, "primary_masks"),
    ("secondary", INTERFACE_SECONDARY_MASKS_KEY, load_interface_secondary_masks, "secondary_masks"),
    ("diffuse_raw", INTERFACE_DIFFUSE_RAW_MASKS_KEY, load_interface_diffuse_raw_masks, "diffuse_raw_masks"),
    ("diffuse", INTERFACE_DIFFUSE_MASKS_KEY, load_interface_diffuse_masks, "diffuse_masks"),
    ("combined", INTERFACE_COMBINED_MASKS_KEY, load_interface_combined_masks, "combined_masks"),
)


def save_specimen(specimen: Specimen, path: JsonLikePath) -> Path:
    """Write the specimen definition (plies, interfaces, metadata) to a JSON file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(specimen.to_dict(), indent=2))
    return target


def _emit(verbose: bool, message: str) -> None:
    if verbose:
        print(message)


def _unique_key(existing: Dict[str, Any], name: str) -> str:
    """Return ``name``, or ``name_2``, ``name_3``, ... if it is already a key of ``existing``."""
    key = str(name)
    if key not in existing:
        return key
    suffix = 2
    while f"{key}_{suffix}" in existing:
        suffix += 1
    return f"{key}_{suffix}"


def load_stored_results(
    specimen: Specimen,
    *,
    strict: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Load every crack and delamination file recorded in the specimen's metadata.

    Parameters
    ----------
    specimen:
        Specimen whose plies and interfaces point to saved results.
    strict:
        Raise if a file can't be loaded. Otherwise the failure is skipped
        (and printed when ``verbose``).
    verbose:
        Print what was found.

    Returns
    -------
    dict[str, Any]
        ``{"plies": {...}, "interfaces": {...}, "summary": [str, ...]}``.
    """
    report: Dict[str, Any] = {"plies": {}, "interfaces": {}, "summary": []}

    def load(loader: Callable[[Any], Any], target: Any, description: str) -> Optional[Any]:
        try:
            return loader(target)
        except Exception as exc:
            message = f"Failed to load {description}: {exc}"
            if strict:
                raise RuntimeError(message) from exc
            _emit(verbose, message)
            return None

    def announce(message: str) -> None:
        report["summary"].append(message)
        _emit(verbose, message)

    for ply in specimen.plies:
        if not ply.metadata.get(PLY_CRACK_RESULTS_KEY):
            continue
        bundle = load(load_ply_crack_results, ply, f"cracks for ply '{ply.name}'")
        if bundle is not None:
            announce(f"Found cracks for ply '{ply.name}' ({len(bundle)} frames).")
            report["plies"][_unique_key(report["plies"], ply.name)] = {"cracks": bundle}

    for interface in specimen.interfaces:
        iface_report: Dict[str, Any] = {}
        found_labels = []

        for label, key, loader, report_key in _INTERFACE_ARTEFACTS:
            if not interface.metadata.get(key):
                continue
            bundle = load(loader, interface, f"{label} delamination for interface '{interface.name}'")
            if bundle is not None:
                iface_report[report_key] = bundle
                found_labels.append(f"{label} ({len(bundle)} frames)")

        metrics_path = interface.metadata.get(INTERFACE_METRICS_KEY)
        if metrics_path:
            metrics = load(load_interface_metrics, interface, f"metrics for interface '{interface.name}'")
            if metrics is not None:
                iface_report["metrics"] = metrics
                iface_report["metrics_path"] = str(Path(metrics_path))
                found_labels.append(f"metrics ({len(metrics)} rows)")

        if found_labels:
            announce(
                f"Found edge/diffuse delamination artefacts for interface "
                f"'{interface.name}': {', '.join(found_labels)}."
            )
            report["interfaces"][_unique_key(report["interfaces"], interface.name)] = iface_report

    return report


def load_specimen(
    path: JsonLikePath,
    *,
    auto_init_stacks: bool = False,
    load_results: bool = False,
    strict: bool = False,
    verbose: bool = False,
) -> Specimen:
    """Rebuild a specimen from a JSON file written by :func:`save_specimen`.

    Parameters
    ----------
    path:
        JSON file to read.
    auto_init_stacks:
        Load the image stacks as well.
    load_results:
        Also load the result files recorded in the metadata, to check they
        are readable.
    strict, verbose:
        Passed to :func:`load_stored_results` when ``load_results`` is set.
    """
    payload = json.loads(Path(path).read_text())
    specimen = Specimen.from_dict(payload, auto_init_stacks=auto_init_stacks)
    if load_results:
        load_stored_results(specimen, strict=strict, verbose=verbose)
    return specimen


__all__ = ["load_specimen", "load_stored_results", "save_specimen"]
