"""Convenience helpers to persist and restore :class:`deladect.specimen.Specimen` objects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar, Union

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
T = TypeVar("T")


def save_specimen(specimen: Specimen, path: JsonLikePath) -> Path:
    """Persist the specimen definition (plies, interfaces, metadata) to JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = specimen.to_dict()
    target.write_text(json.dumps(payload, indent=2))
    return target


def _emit(verbose: bool, message: str) -> None:
    if verbose:
        print(message)


def _safe_bundle_load(
    *,
    loader: Callable[[], T],
    description: str,
    strict: bool,
    verbose: bool,
) -> Optional[T]:
    try:
        bundle = loader()
    except Exception as exc:
        message = f"Failed to load {description}: {exc}"
        if strict:
            raise RuntimeError(message) from exc
        _emit(verbose, message)
        return None
    return bundle


def _unique_key(existing: Dict[str, Any], name: str) -> str:
    """Create a stable key, suffixing duplicates as ``<name>_<n>``."""
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
    """Load all crack and delamination artefacts referenced in specimen metadata.

    Parameters
    ----------
    specimen:
        Specimen whose ply/interface metadata points to persisted artefacts.
    strict:
        If ``True``, raise when a referenced artefact cannot be loaded.
        If ``False``, keep going and report failures in ``summary``.
    verbose:
        If ``True``, print human-readable discovery/loading messages.

    Returns
    -------
    dict[str, Any]
        Nested dictionary with loaded bundles and a summary list.
    """
    report: Dict[str, Any] = {
        "plies": {},
        "interfaces": {},
        "summary": [],
    }

    for ply in specimen.plies:
        ply_report: Dict[str, Any] = {}
        if ply.metadata.get(PLY_CRACK_RESULTS_KEY):
            bundle = _safe_bundle_load(
                loader=lambda: load_ply_crack_results(ply),
                description=f"cracks for ply '{ply.name}'",
                strict=strict,
                verbose=verbose,
            )
            if bundle is not None:
                ply_report["cracks"] = bundle
                msg = f"Found cracks for ply '{ply.name}' ({len(bundle)} frames)."
                report["summary"].append(msg)
                _emit(verbose, msg)
        if ply_report:
            report["plies"][_unique_key(report["plies"], ply.name)] = ply_report

    interface_loaders = (
        ("edge", INTERFACE_PRIMARY_MASKS_KEY, load_interface_primary_masks, "primary_masks"),
        ("secondary", INTERFACE_SECONDARY_MASKS_KEY, load_interface_secondary_masks, "secondary_masks"),
        ("diffuse_raw", INTERFACE_DIFFUSE_RAW_MASKS_KEY, load_interface_diffuse_raw_masks, "diffuse_raw_masks"),
        ("diffuse", INTERFACE_DIFFUSE_MASKS_KEY, load_interface_diffuse_masks, "diffuse_masks"),
        ("combined", INTERFACE_COMBINED_MASKS_KEY, load_interface_combined_masks, "combined_masks"),
    )

    for interface in specimen.interfaces:
        iface_report: Dict[str, Any] = {}
        found_labels = []

        for label, key, loader, report_key in interface_loaders:
            if not interface.metadata.get(key):
                continue
            bundle = _safe_bundle_load(
                loader=lambda loader=loader, interface=interface: loader(interface),
                description=f"{label} delamination for interface '{interface.name}'",
                strict=strict,
                verbose=verbose,
            )
            if bundle is None:
                continue
            iface_report[report_key] = bundle
            found_labels.append(f"{label} ({len(bundle)} frames)")

        metrics_path = interface.metadata.get(INTERFACE_METRICS_KEY)
        if metrics_path:
            metrics = _safe_bundle_load(
                loader=lambda interface=interface: load_interface_metrics(interface),
                description=f"metrics for interface '{interface.name}'",
                strict=strict,
                verbose=verbose,
            )
            if metrics is not None:
                iface_report["metrics"] = metrics
                iface_report["metrics_path"] = str(Path(metrics_path))
                found_labels.append(f"metrics ({len(metrics)} rows)")

        if found_labels:
            msg = (
                f"Found edge/diffuse delamination artefacts for interface "
                f"'{interface.name}': {', '.join(found_labels)}."
            )
            report["summary"].append(msg)
            _emit(verbose, msg)
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
    """Rebuild a specimen from a previously saved JSON snapshot.

    Parameters
    ----------
    path:
        Path to a JSON snapshot previously produced by :func:`save_specimen`.
    auto_init_stacks:
        If ``True``, initialize crackdect image stacks during reconstruction.
    load_results:
        If ``True``, eagerly load artefacts referenced in ply/interface metadata
        and emit discovery messages when ``verbose=True``.
    strict:
        If ``True`` and ``load_results`` is enabled, raise when a referenced
        artefact cannot be loaded.
    verbose:
        If ``True`` and ``load_results`` is enabled, print discovery messages.
    """
    source = Path(path)
    payload = json.loads(source.read_text())
    specimen = Specimen.from_dict(payload, auto_init_stacks=auto_init_stacks)
    if load_results:
        load_stored_results(specimen, strict=strict, verbose=verbose)
    return specimen


__all__ = ["load_specimen", "load_stored_results", "save_specimen"]
