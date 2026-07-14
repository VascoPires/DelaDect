"""Specimen, ply, and interface classes used throughout DelaDect.

This module provides a central Specimen class and keeps the related data structures in one place. For clarity (and for visualization), 
it is recommended to define plies and interfaces in the same order as they are stacked in the real specimen, although this is not strictly required.

As a general rule, the classes are intended to be used as follows:

* A specimen is an assembly of plies and interfaces. And contains the most relevant
  metadata for the specimen, namely the image stacks.
* A Ply corresponds to the entity used for crack detection. This means that
  any crack detection goes through the ply class. A direction must be defined for each ply,
  which is used to detect cracks in that specifinoc direction. Due to the nature of the method,
  even if multiple plies have the same direction, the method will not be able to distinguish between them.
  So, for repeated plies, only one crack detection is performed and reported.
* Interfaces correspond exclusively to the object related with delamination detection. For multi-edge
  delamination detection, multiple interfaces need be defined.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import warnings

import numpy as np
import pandas as pd
from skimage.io import imread
from pathlib import Path


# Test if main requirements are installed
try:  
    from crackdect import ImageStack, ImageStackSQL, image_paths, sort_paths
except Exception as exc: 
    ImageStack = ImageStackSQL = None  # type: ignore[assignment]
    image_paths = sort_paths = None  # type: ignore[assignment]
    _CRACKDECT_IMPORT_ERROR = exc
else:
    _CRACKDECT_IMPORT_ERROR = None  # type: ignore[assignment]

Color = Tuple[float, float, float, float]

logger = logging.getLogger(__name__)


def rgba_from_hex(hex_color: str, alpha: float = 1.0) -> Color:
    """Convert ``#RRGGBB`` + alpha into an RGBA tuple (each entry 0-1).
    This is just a helper function to define colours for plies and interfaces.
    """
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError("hex_color must be in the form #RRGGBB")
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return (r, g, b, alpha)


DEFAULT_PLY_COLOR: Color = rgba_from_hex("#888888", 0.3)
DEFAULT_CRACK_COLOR: Color = rgba_from_hex("#E74C3C", 0.85)
DEFAULT_PRIMARY_DELAMINATION_COLOR: Color = rgba_from_hex("#E53935", 0.9)
DEFAULT_SECONDARY_DELAMINATION_COLOR: Color = rgba_from_hex("#1E88E5", 0.75)


if sys.version_info >= (3, 10):
    _dataclass = dataclass
    _dataclass_kwargs = {"slots": True}
else:
    _dataclass = dataclass
    _dataclass_kwargs = {}


@_dataclass(**_dataclass_kwargs)
class Ply:
    """Metadata for a single ply/layer in the laminate.

    A `Ply` is the unit used for crack detection. The `orientation_deg` defines
    which crack direction is targeted for this ply.

    Attributes
    ----------
    name:
        Human-readable ply name (used in reports and plots).
    orientation_deg:
        Ply orientation in degrees. This is used to select the crack direction
        to be detected for this ply.
    avg_crack_width_px:
        Expected average crack width in pixels. Used as a tuning parameter for
        crack detection.
    min_crack_length_px:
        Minimum crack length in pixels for a detected feature to be reported.
    thickness_mm:
        Optional ply thickness in millimeters used by 3D visualizations.
        Defaults to ``1.0`` mm when not provided.
    color_rgba:
        RGBA color used to visualize this ply in plots.
    crack_color_rgba:
        RGBA color used to visualize cracks associated with this ply.
    metadata:
        Free-form dictionary for extra user-defined information.
    

    Example
    -------
    >>> Ply(
    ...     name="plus45",
    ...     orientation_deg=45.0,
    ...     avg_crack_width_px=8.0,
    ...     min_crack_length_px=20.0,
    ... )
    """

    name: str
    orientation_deg: float
    avg_crack_width_px: float
    min_crack_length_px: float
    thickness_mm: float = 1.0
    color_rgba: Color = DEFAULT_PLY_COLOR
    crack_color_rgba: Color = DEFAULT_CRACK_COLOR
    metadata: Dict[str, Any] = field(default_factory=dict)


@_dataclass(**_dataclass_kwargs)
class Interface:
    """Description of a delamination interface between two plies.

    An `Interface` represents a potential delamination plane in the laminate.
    Optionally, it can be linked to the plies above and below via their indices.

    Attributes
    ----------
    name:
        Name of the interface, e.g. `"0/90"`.
    upper_ply_index:
        Index of the ply above this interface in the specimen ply list. If `None`,
        the interface is not explicitly linked to a specific ply.
    lower_ply_index:
        Index of the ply below this interface in the specimen ply list. If `None`,
        the interface is not explicitly linked to a specific ply.
    enabled:
        If `False`, this interface is ignored during delamination detection.
    delamination_color_rgba:
        RGBA color used to visualize delaminations associated with this interface.
    metadata:
        Free-form dictionary for extra user-defined information. Deladect already uses
        by default to save some metadata such as previous saved results and delamination
        masks.

    Example
    -------
    >>> Interface(name="0/90")
    """

    name: str
    upper_ply_index: Optional[int] = None
    lower_ply_index: Optional[int] = None
    enabled: bool = True
    delamination_color_rgba: Color = DEFAULT_PRIMARY_DELAMINATION_COLOR
    metadata: Dict[str, Any] = field(default_factory=dict)


@_dataclass(**_dataclass_kwargs)
class Specimen:
    """Specimen class used to manage crack/delamination workflows for a single laminate specimen.

    The class centralizes specimen metadata, image stacks, ply/interface definitions,
    and export-related helpers. Detection algorithms are implemented in
    `deladect.detection` and use the `Specimen` class as input.

    Args:
        name: Identifier for the specimen (used in filenames and manifests).
        scale_px_mm: Conversion factor from millimetres to pixels (px/mm).
        path_full: Directory containing the full specimen view frames (required).
        path_upper_border: Directory with images describing the upper border (optional).
        path_lower_border: Directory with images describing the lower border (optional).
        path_middle: Directory with frames of the middle region used for crack analysis (optional).
        sorting_key: Key passed to :func:`crackdect.sort_paths` to order image stacks.
        image_types: Iterable of image suffixes/extensions to include (e.g. ``[".png"]``).
        avg_crack_width_px: Nominal average crack width in pixels for the specimen .
        dimensions: Optional mapping with geometric info in millimeters. Accepted keys include
            ``width_mm``/``width``, ``height_mm``/``length_mm``/``height``/``length``,
            and ``thickness_mm``/``thickness``.
        strain_csv: Optional CSV file containing a ``strain_y`` column to merge later.
        stack_backend: ``"auto"``, ``"memory"`` or ``"sql"`` choice for stack storage.
        stack_limit_mb: Memory ceiling (MB) before ``"auto"`` flips to SQL-backed stacks.
        sql_stack_kwargs: Extra keyword arguments forwarded to :meth:`ImageStackSQL.from_paths`.
        plies: Optional list of pre-defined :class:`Ply` records.
        interfaces: Optional list of :class:`Interface` entries.
        crack_color_rgba: Default crack colour applied when a ply doesn't override it.

    Raises:
        ValueError: If ``stack_backend`` is not recognised.

    Example
    -------
    >>> specimen = Specimen(
    ...     name="sample_01",
    ...     scale_px_mm=35.0,
    ...     path_full="data/sample_01/full",
    ...     path_upper_border=None,
    ...     path_lower_border=None,so 
    ...     path_middle=None,
    ...     sorting_key="frame_idx",
    ...     image_types=[".png"],
    ... )
    >>> specimen.add_ply(name="plus45", orientation_deg=45.0)
    >>> specimen.add_interface(name="top_interface", upper_ply_index=0, lower_ply_index=0)
    """

    name: str
    scale_px_mm: float
    path_full: str
    sorting_key: str
    image_types: List[str]
    path_upper_border: Optional[str] = None
    path_lower_border: Optional[str] = None
    path_middle: Optional[str] = None
    avg_crack_width_px: float = 10.0
    dimensions: Optional[Dict[str, float]] = None
    strain_csv: Optional[str] = None
    stack_backend: str = "auto"
    stack_limit_mb: float = 2048.0
    sql_stack_kwargs: Optional[Dict[str, Any]] = None
    results_root: Optional[str] = None
    plies: List[Ply] = field(default_factory=list)
    interfaces: List[Interface] = field(default_factory=list)
    crack_color_rgba: Color = DEFAULT_CRACK_COLOR
    experimental_data: Optional[pd.DataFrame] = field(init=False, default=None)
    _stack_backend: str = field(init=False, default="auto")
    _stack_limit_bytes: float = field(init=False, default=0.0)
    _sql_stack_kwargs: Dict[str, Any] = field(init=False, default_factory=dict)
    _image_types_normalized: List[str] = field(init=False, default_factory=list)
    _results_root: Path = field(init=False)
    auto_init_stacks: bool = True
    path_full_list: List[str] = field(init=False, default_factory=list)
    path_upper_list: List[str] = field(init=False, default_factory=list)
    path_lower_list: List[str] = field(init=False, default_factory=list)
    path_middle_list: List[str] = field(init=False, default_factory=list)
    image_stack_full: Optional[Any] = field(init=False, default=None)
    image_stack_upper: Optional[Any] = field(init=False, default=None)
    image_stack_lower: Optional[Any] = field(init=False, default=None)
    image_stack_middle: Optional[Any] = field(init=False, default=None)

    def __post_init__(self) -> None:
        """Validate the stack configuration and load optional metadata.

        Construction is intentionally in-memory: creating a :class:`Specimen`
        never creates result folders or writes a configuration file.  Call
        :meth:`save_config` when the specimen definition is complete.
        """
        backend = (self.stack_backend or "auto").lower()
        if backend not in {"auto", "memory", "sql"}:
            raise ValueError('stack_backend must be "auto", "memory", or "sql"')
        self._stack_backend = backend
        self._stack_limit_bytes = max(float(self.stack_limit_mb), 0.0) * 1024 * 1024
        self._sql_stack_kwargs = dict(self.sql_stack_kwargs or {})
        self._image_types_normalized = self._normalize_image_types(self.image_types)
        self._results_root = self._build_results_root(self.results_root)

        if self.auto_init_stacks:
            self._initialize_image_stacks()
            self._emit_region_message()

        if self.strain_csv is not None:
            df = pd.read_csv(self.strain_csv)
            self.experimental_data = df[["strain_y"]].reset_index(drop=True)

        self._normalize_interface_indices()

    def _build_results_root(self, results_root: Optional[str]) -> Path:
        """Builds the the specimen results root path. This is just a pure path
        builder. To not be confused with the other results methods.
        """
        if results_root is None:
            configured = self.results_root
            base = Path("results").resolve() if configured is None else Path(configured).resolve()
        else:
            base = Path(results_root).resolve()
        specimen_name = self._validate_result_component(self.name, label="specimen name")
        if base.name != specimen_name:
            base = base / specimen_name
        return base

    @staticmethod
    def _validate_result_component(value: str, *, label: str) -> str:
        """Return one safe relative directory component.

        Result-directory components are identifiers, not arbitrary paths.  In
        particular, accepting ``..`` or an absolute path would allow callers
        to write outside the specimen's configured result root.
        """
        component = str(value).strip()
        path = Path(component)
        if (
            not component
            or path.is_absolute()
            or path.drive
            or component in {".", ".."}
            or len(path.parts) != 1
        ):
            raise ValueError(
                f"{label} must be one non-empty relative path component; got {value!r}."
            )
        return component

    def resolve_results_root(self, results_root: Optional[str] = None) -> Path:
        """Returns the base results folder for this specimen. In this folder
        all the results related to this specimen will be stored. 
        If the folder does not exist, it is created."""
        base = self._build_results_root(results_root)
        base.mkdir(parents=True, exist_ok=True)
        return base
    
    def results_dir(self, *parts: str, results_root: Optional[str] = None) -> Path:
        """Return a writable directory contained in this specimen's result root.

        Each ``parts`` entry must be a single relative directory component.
        Use ``results_root`` to intentionally select a different output root.
        """
        base = self.resolve_results_root(results_root)
        for part in parts:
            if part:
                base /= self._validate_result_component(part, label="results directory part")
        base.mkdir(parents=True, exist_ok=True)
        return base

    def results_root_path(self) -> Path:
        """Return the base results folder for this specimen."""
        return Path(self._results_root)

    def config_path(self) -> Path:
        """Return the default configuration path under results/config."""
        config_dir = self.results_dir("config")
        name = self._validate_result_component(self.name, label="specimen name")
        return config_dir / f"{name}_config.json"

    def save_config(self) -> Path:
        """Persist the current specimen configuration under results/config.

        This is explicit by design: mutations such as :meth:`add_ply` and
        :meth:`add_interface` are not silently written to disk.
        """
        from deladect.io.specimen_io import save_specimen

        target = self.config_path()
        save_specimen(self, target)
        return target

    def _emit_region_message(self) -> None:
        """Log which stack regions are available for analysis.
           This is only relevant if the user wants to use manually
           include slices of the image instead of providing a full
           image. However, if one of the upper, lower or middle regions
           are provided, a warming message is shown if the user did not fully
           define all of them.
           
        """
        if any((self.path_upper_border, self.path_lower_border, self.path_middle)):
            missing = []
            if not self.path_upper_border:
                missing.append("upper")
            if not self.path_lower_border:
                missing.append("lower")
            if not self.path_middle:
                missing.append("middle")
            message = "Performing analysis with manual overwritten images."
            if missing:
                message += f" Missing regions: {', '.join(missing)}."
            else:
                message += " All regions were loaded."
            logger.debug(message)
        else:
            logger.debug(
                "No region overrides provided; crack evaluation will use the full specimen stack."
            )

    @staticmethod
    def _estimate_stack_bytes(
        paths: List[str],
        *,
        dtype: Any = np.float32,
        as_gray: Optional[bool] = True,
    ) -> int:
        """This function estimates the memory footprint (bytes) for loading ``paths``.
        Based on the estimated memory and the default value of ``stack_limit_mb``, the stack backend 
        is selected when ``stack_backend="auto"``.
        """
        paths = list(paths)
        if not paths:
            return 0
        sample = imread(paths[0], as_gray=as_gray) if as_gray is not None else imread(paths[0])
        arr = np.asarray(sample, dtype=dtype) if dtype is not None else np.asarray(sample)
        return int(arr.nbytes) * len(paths)

    def _build_stack(
        self,
        paths: List[str],
        *,
        dtype: Any = np.float32,
        as_gray: Optional[bool] = True,
    ):
        """Build an ImageStack or ImageStackSQL instance based on the backend."""
        if _CRACKDECT_IMPORT_ERROR is not None:
            raise RuntimeError(
                "crackdect is required to build image stacks; install crackdect before instantiating Specimen."
            ) from _CRACKDECT_IMPORT_ERROR

        if ImageStack is None or ImageStackSQL is None:
            raise RuntimeError("crackdect backends are unavailable.")

        paths = list(paths)
        if not paths:
            raise ValueError("Cannot build an image stack without any image paths.")

        selected = self._stack_backend
        if selected == "auto":
            est_bytes = self._estimate_stack_bytes(paths, dtype=dtype, as_gray=as_gray)
            if self._stack_limit_bytes > 0 and est_bytes > self._stack_limit_bytes:
                selected = "sql"
            else:
                selected = "memory"

        if selected == "sql":
            kwargs = dict(self._sql_stack_kwargs)
            if dtype is not None and "dtype" not in kwargs:
                kwargs["dtype"] = dtype
            if as_gray is not None and "as_gray" not in kwargs:
                kwargs["as_gray"] = as_gray
            return ImageStackSQL.from_paths(paths, **kwargs)

        if selected != "memory":
            raise ValueError(f"Unsupported stack backend '{selected}'.")

        kwargs = {}
        if dtype is not None:
            kwargs["dtype"] = dtype
        if as_gray is not None:
            kwargs["as_gray"] = as_gray
        return ImageStack.from_paths(paths, **kwargs)

    @staticmethod
    def _normalize_image_types(image_types: Iterable[str]) -> List[str]:
        """Normalize extensions to lowercase values without leading dots.
        
        ### Example:
        [".PNG", " jpg ", "bmp", "", ".TIF"]
        -> ["png", "jpg", "bmp", "tif"]

        """
        normalized: List[str] = []
        for ext in image_types:
            cleaned = str(ext).strip().lower()
            if not cleaned:
                continue
            normalized.append(cleaned.lstrip("."))
        return normalized

    def _initialize_image_stacks(self) -> None:
        """Load the specimen's image stacks for full/upper/lower/middle regions."""
        if _CRACKDECT_IMPORT_ERROR is not None:
            # Defer raising until someone actually initialises image stacks.
            raise RuntimeError(
                "crackdect is required to initialise specimen image stacks. "
                "Install crackdect or override `_initialize_image_stacks`."
            ) from _CRACKDECT_IMPORT_ERROR

        region_specs = (
            ("full", self.path_full, np.float32, True),
            ("upper", self.path_upper_border, np.float32, True),
            ("lower", self.path_lower_border, np.float32, True),
            ("middle", self.path_middle, np.float32, True),
        )

        for name, folder, dtype, as_gray in region_specs:
            if folder is None:
                if name == "full":
                    raise ValueError("Specimen requires a valid path_full to initialise the image stack.")
                setattr(self, f"path_{name}_list", [])
                setattr(self, f"image_stack_{name}", None)
                continue
            self._load_region_stack(
                name=name,
                folder=folder,
                dtype=dtype,
                as_gray=as_gray,
            )

    def _load_region_stack(
        self,
        *,
        name: str,
        folder: str,
        dtype: Any,
        as_gray: Optional[bool],
    ) -> None:
        """Create an ImageStack for a single specimen region and attach it."""
        if image_paths is None or sort_paths is None:
            raise RuntimeError("crackdect helpers are required to discover image paths.")

        normalized_types = self._image_types_normalized or ["png", "jpg", "bmp"]
        paths = list(image_paths(folder, image_types=normalized_types))
        if not paths:
            raise ValueError(f"No images found for region {name!r} in {folder!r}.")

        sorted_paths, _ = sort_paths(paths, sorting_key=self.sorting_key)
        if sorted_paths.size == 0:
            paths_list = sorted(map(str, paths))
        else:
            paths_list = [str(p) for p in sorted_paths]
        setattr(self, f"path_{name}_list", paths_list)

        stack = self._build_stack(paths_list, dtype=dtype, as_gray=as_gray)
        setattr(self, f"image_stack_{name}", stack)

    # ------------------------------------------------------------------
    # Serialization helpers (config + metadata paths)
    # ------------------------------------------------------------------
    # These helpers persist/rebuild the specimen definition itself.
    # Heavy artefacts (crack bundles, delamination masks) are stored separately
    # as NPZ/CSV files; only their paths are serialized via ply/interface metadata.

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable snapshot of this specimen.

        The payload includes specimen configuration, ply/interface definitions,
        and metadata dictionaries. Image-stack arrays are not embedded in JSON.
        """
        return {
            "name": self.name,
            "scale_px_mm": self.scale_px_mm,
            "path_full": self.path_full,
            "path_upper_border": self.path_upper_border,
            "path_lower_border": self.path_lower_border,
            "path_middle": self.path_middle,
            "sorting_key": self.sorting_key,
            "image_types": self.image_types,
            "avg_crack_width_px": self.avg_crack_width_px,
            "dimensions": self.dimensions,
            "strain_csv": self.strain_csv,
            "stack_backend": self.stack_backend,
            "stack_limit_mb": self.stack_limit_mb,
            "sql_stack_kwargs": self.sql_stack_kwargs,
            "results_root": self.results_root,
            "plies": [self._ply_to_dict(ply) for ply in self.plies],
            "interfaces": [self._interface_to_dict(interface) for interface in self.interfaces],
            "crack_color_rgba": list(self.crack_color_rgba),
            "auto_init_stacks": self.auto_init_stacks,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any], *, auto_init_stacks: Optional[bool] = None) -> "Specimen":
        """Rebuild a specimen from :meth:`to_dict` output.

        Notes
        -----
        - Ply/interface metadata is restored as stored.
        - If ``auto_init_stacks`` is provided, it overrides the serialized value.
        """
        plies_payload = payload.get("plies", [])
        interfaces_payload = payload.get("interfaces", [])
        specimen_kwargs = dict(payload)
        specimen_kwargs["plies"] = [cls._ply_from_dict(data) for data in plies_payload]
        specimen_kwargs["interfaces"] = [cls._interface_from_dict(data) for data in interfaces_payload]
        specimen_kwargs["crack_color_rgba"] = tuple(specimen_kwargs.get("crack_color_rgba", DEFAULT_CRACK_COLOR))
        specimen_kwargs["auto_init_stacks"] = specimen_kwargs.get("auto_init_stacks", True)
        if auto_init_stacks is not None:
            specimen_kwargs["auto_init_stacks"] = auto_init_stacks
        return cls(**specimen_kwargs)

    @staticmethod
    def _ply_to_dict(ply: Ply) -> Dict[str, Any]:
        """Serialize a :class:`Ply` into JSON-compatible primitives."""
        return {
            "name": ply.name,
            "orientation_deg": ply.orientation_deg,
            "avg_crack_width_px": ply.avg_crack_width_px,
            "min_crack_length_px": ply.min_crack_length_px,
            "thickness_mm": ply.thickness_mm,
            "color_rgba": list(ply.color_rgba),
            "crack_color_rgba": list(ply.crack_color_rgba),
            "metadata": ply.metadata,
        }

    @staticmethod
    def _ply_from_dict(payload: Dict[str, Any]) -> Ply:
        """Deserialize a :class:`Ply` from :meth:`_ply_to_dict` output.

        Missing optional fields fall back to module defaults.
        """
        return Ply(
            name=payload["name"],
            orientation_deg=payload["orientation_deg"],
            avg_crack_width_px=payload["avg_crack_width_px"],
            min_crack_length_px=payload["min_crack_length_px"],
            thickness_mm=float(payload.get("thickness_mm", 1.0)),
            color_rgba=tuple(payload.get("color_rgba", DEFAULT_PLY_COLOR)),
            crack_color_rgba=tuple(payload.get("crack_color_rgba", DEFAULT_CRACK_COLOR)),
            metadata=payload.get("metadata", {}),
        )

    @staticmethod
    def _interface_to_dict(interface: Interface) -> Dict[str, Any]:
        """Serialize an :class:`Interface` into JSON-compatible primitives."""
        return {
            "name": interface.name,
            "upper_ply_index": interface.upper_ply_index,
            "lower_ply_index": interface.lower_ply_index,
            "enabled": interface.enabled,
            "delamination_color_rgba": list(interface.delamination_color_rgba),
            "metadata": interface.metadata,
        }

    @staticmethod
    def _interface_from_dict(payload: Dict[str, Any]) -> Interface:
        """Deserialize an :class:`Interface` from :meth:`_interface_to_dict` output.

        Missing optional fields fall back to module defaults.
        """
        return Interface(
            name=payload["name"],
            upper_ply_index=payload.get("upper_ply_index"),
            lower_ply_index=payload.get("lower_ply_index"),
            enabled=payload.get("enabled", True),
            delamination_color_rgba=tuple(
                payload.get("delamination_color_rgba", DEFAULT_PRIMARY_DELAMINATION_COLOR)
            ),
            metadata=payload.get("metadata", {}),
        )

    # ------------------------------------------------------------------
    # Ply helpers. 
    # Below some ply related functions for adding, removing and other ply functionalities.
    # ------------------------------------------------------------------

    def add_ply(
        self,
        ply: Optional[Ply] = None,
        *,
        name: Optional[str] = None,
        orientation_deg: Optional[float] = None,
        avg_crack_width_px: Optional[float] = None,
        min_crack_length_px: Optional[float] = None,
        thickness_mm: Optional[float] = None,
        color_rgba: Optional[Color] = None,
        crack_color_rgba: Optional[Color] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Ply:
        """Appends/adds a ply to a given specimen. If a Ply instance is not provided, 
        a new one will be constructed from the keyword arguments.

        Example
        -------
        >>> specimen.add_ply(name=\"plus45\", orientation_deg=45.0)
        """
        if ply is None:
            if name is None or orientation_deg is None:
                raise ValueError("`name` and `orientation_deg` are required when creating a ply.")
            avg_width = avg_crack_width_px if avg_crack_width_px is not None else self.avg_crack_width_px
            min_length = (
                min_crack_length_px
                if min_crack_length_px is not None
                else max(avg_width * 2.0, avg_width)
            )
            ply = Ply(
                name=name,
                orientation_deg=orientation_deg,
                avg_crack_width_px=avg_width,
                min_crack_length_px=min_length,
                thickness_mm=float(thickness_mm) if thickness_mm is not None else 1.0,
                color_rgba=color_rgba or DEFAULT_PLY_COLOR,
                crack_color_rgba=crack_color_rgba or self.crack_color_rgba,
                metadata=metadata or {},
            )
        self.plies.append(ply)
        return ply

    def remove_ply(self, *, name: Optional[str] = None, index: Optional[int] = None) -> Ply:
        """Remove a ply by ``name`` or ``index``.

        Firesafe behaviour: interfaces that reference the removed ply are marked
        for review (indices left partially unresolved and ``enabled=False``), and
        a runtime warning is emitted.
        """
        if name is None and index is None:
            raise ValueError("Provide either `name` or `index` when removing a ply.")

        target_index: Optional[int] = None
        if name is not None:
            for idx, candidate in enumerate(self.plies):
                if candidate.name == name:
                    target_index = idx
                    break
            if target_index is None:
                raise ValueError(f"No ply named '{name}' found.")
        elif index is not None:
            target_index = int(index)
            if target_index < 0:
                target_index += len(self.plies)
            if target_index < 0 or target_index >= len(self.plies):
                raise IndexError("ply index out of range")

        if target_index is None:
            raise ValueError("Invalid remove request.")

        removed = self.plies.pop(target_index)
        affected_interfaces = self._apply_ply_removal_to_interfaces(target_index)
        if affected_interfaces:
            warnings.warn(
                "Removed ply affects interface mapping. The following interfaces were "
                "disabled and require index review: "
                + ", ".join(affected_interfaces),
                RuntimeWarning,
                stacklevel=2,
            )
        return removed

    def _apply_ply_removal_to_interfaces(self, removed_index: int) -> List[str]:
        """Update interface indices after a ply removal and flag affected entries."""
        affected: List[str] = []
        for interface in self.interfaces:
            upper = interface.upper_ply_index
            lower = interface.lower_ply_index
            touched_removed = False

            if upper is not None:
                if upper == removed_index:
                    upper = None
                    touched_removed = True
                elif upper > removed_index:
                    upper -= 1

            if lower is not None:
                if lower == removed_index:
                    lower = None
                    touched_removed = True
                elif lower > removed_index:
                    lower -= 1

            interface.upper_ply_index = upper
            interface.lower_ply_index = lower

            if touched_removed:
                interface.enabled = False
                interface.metadata["index_review_required"] = True
                affected.append(interface.name)
            elif upper is not None and lower is not None:
                interface.metadata.pop("index_review_required", None)

        return affected

    def get_ply_by_name(self, name: str) -> Optional[Ply]:
        """Return the first ply matching ``name`` (handy for CLI hooks)."""
        return next((ply for ply in self.plies if ply.name == name), None)

    def get_ply_by_orientation(self, orientation_deg: float, *, tolerance: float = 1e-3) -> Optional[Ply]:
        """Return the first ply whose orientation matches within ``tolerance`` degrees."""
        return next(
            (ply for ply in self.plies if abs(ply.orientation_deg - orientation_deg) <= tolerance),
            None,
        )

    def get_plies_by_orientation(self, orientation_deg: float, *, tolerance: float = 1e-3) -> List[Ply]:
        """Return all plies whose orientation matches within ``tolerance`` degrees."""
        return [ply for ply in self.plies if abs(ply.orientation_deg - orientation_deg) <= tolerance]

    def iter_plies(self) -> Iterable[Ply]:
        """Yield plies in their current stacking order."""
        yield from self.plies

    # ------------------------------------------------------------------
    # Interface helpers
    # ------------------------------------------------------------------

    def add_interface(
        self,
        interface: Optional[Interface] = None,
        *,
        name: Optional[str] = None,
        upper_ply_index: Optional[int] = None,
        lower_ply_index: Optional[int] = None,
        enabled: bool = True,
        delamination_color_rgba: Optional[Color] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Interface:
        """Append an interface, optionally constructing one from keyword arguments.

        When ply indices are omitted, DelaDect applies firesafe inference:

        - both omitted: infer by interface order (``0/1``, ``1/2``, ...)
        - one omitted: infer adjacent ply (``upper = lower - 1`` or
          ``lower = upper + 1``)

        Example
        -------
        >>> specimen.add_interface(name=\"0/90\", upper_ply_index=0, lower_ply_index=1)
        >>> specimen.add_interface(name=\"top_interface\")
        """
        if interface is None:
            if name is None:
                raise ValueError("name is required for new interfaces.")
            interface = Interface(
                name=name,
                upper_ply_index=upper_ply_index,
                lower_ply_index=lower_ply_index,
                enabled=enabled,
                delamination_color_rgba=delamination_color_rgba or DEFAULT_PRIMARY_DELAMINATION_COLOR,
                metadata=metadata or {},
            )

        resolved_upper, resolved_lower = self._resolve_interface_indices(
            upper_ply_index=interface.upper_ply_index,
            lower_ply_index=interface.lower_ply_index,
            interface_position=len(self.interfaces),
            interface_name=interface.name,
        )
        interface.upper_ply_index = resolved_upper
        interface.lower_ply_index = resolved_lower

        self.interfaces.append(interface)
        return interface

    def _normalize_interface_indices(self) -> None:
        """Normalize and validate existing interface ply indices."""
        for idx, interface in enumerate(self.interfaces):
            allow_inference = not bool(interface.metadata.get("index_review_required", False))
            upper, lower = self._resolve_interface_indices(
                upper_ply_index=interface.upper_ply_index,
                lower_ply_index=interface.lower_ply_index,
                interface_position=idx,
                interface_name=interface.name,
                allow_inference=allow_inference,
            )
            interface.upper_ply_index = upper
            interface.lower_ply_index = lower

    def _resolve_interface_indices(
        self,
        *,
        upper_ply_index: Optional[int],
        lower_ply_index: Optional[int],
        interface_position: int,
        interface_name: str,
        allow_inference: bool = True,
    ) -> Tuple[Optional[int], Optional[int]]:
        """Infer and validate ply indices for one interface."""
        ply_count = len(self.plies)

        upper = None if upper_ply_index is None else int(upper_ply_index)
        lower = None if lower_ply_index is None else int(lower_ply_index)

        if not allow_inference:
            if ply_count > 0:
                if upper is not None and not (0 <= upper < ply_count):
                    raise ValueError(
                        f"Interface '{interface_name}' upper_ply_index={upper} is out of range "
                        f"for {ply_count} plies."
                    )
                if lower is not None and not (0 <= lower < ply_count):
                    raise ValueError(
                        f"Interface '{interface_name}' lower_ply_index={lower} is out of range "
                        f"for {ply_count} plies."
                    )
            return upper, lower

        if upper is None and lower is None:
            if ply_count >= 2 and interface_position < (ply_count - 1):
                return interface_position, interface_position + 1
            if ply_count < 2:
                logger.debug(
                    "Interface '%s' has unresolved ply indices with fewer than 2 plies defined.",
                    interface_name,
                )
                return None, None
            raise ValueError(
                f"Cannot infer ply indices for interface '{interface_name}'. "
                f"Defined interfaces exceed adjacent ply pairs ({ply_count - 1})."
            )

        if upper is None and lower is not None:
            upper = lower - 1
        elif lower is None and upper is not None:
            lower = upper + 1

        if upper is None or lower is None:
            return upper, lower

        if ply_count > 0:
            if not (0 <= upper < ply_count):
                raise ValueError(
                    f"Interface '{interface_name}' upper_ply_index={upper} is out of range "
                    f"for {ply_count} plies."
                )
            if not (0 <= lower < ply_count):
                raise ValueError(
                    f"Interface '{interface_name}' lower_ply_index={lower} is out of range "
                    f"for {ply_count} plies."
                )

        return upper, lower

    def remove_interface(self, *, name: Optional[str] = None, index: Optional[int] = None) -> Interface:
        """Remove an interface by ``name`` or ``index``."""
        if name is None and index is None:
            raise ValueError("Provide either `name` or `index` when removing an interface.")
        if name is not None:
            for idx, candidate in enumerate(self.interfaces):
                if candidate.name == name:
                    return self.interfaces.pop(idx)
            raise ValueError(f"No interface named '{name}' found.")
        if index is not None:
            return self.interfaces.pop(index)
        raise ValueError("Invalid remove request.")

    def get_interface_by_name(self, name: str) -> Optional[Interface]:
        """Return the first interface matching ``name``."""
        return next((interface for interface in self.interfaces if interface.name == name), None)

    def iter_interfaces(self, *, enabled_only: bool = False) -> Iterable[Interface]:
        """Yield interfaces, optionally filtering only the ones marked as ``enabled``."""
        if not enabled_only:
            yield from self.interfaces
        else:
            for interface in self.interfaces:
                if interface.enabled:
                    yield interface

    def upload_experimental_data(
        self,
        data_path: str,
        *,
        sheet_name: Optional[str] = None,
        n0: int = 0,
        nf: Optional[int] = None,
        nstep: int = 1,
    ) -> pd.DataFrame:
        """Load experimental strain data from CSV/Excel and persist it on the specimen."""
        if data_path.lower().endswith(".csv"):
            df = pd.read_csv(data_path)
        else:
            df = pd.read_excel(data_path, sheet_name=sheet_name)
        df_filtered = df.loc[n0:nf:nstep, ["strain_y"]].reset_index(drop=True)
        self.experimental_data = df_filtered
        return df_filtered

    @staticmethod
    def join_cracks(*crack_lists: List[np.ndarray]) -> List[np.ndarray]:
        """Join multiple crack lists frame by frame."""
        if not crack_lists:
            return []
        frame_count = len(crack_lists[0])
        for crack_list in crack_lists:
            if len(crack_list) != frame_count:
                raise ValueError("All crack lists must have the same number of frames.")
        joined: List[np.ndarray] = []
        for idx in range(frame_count):
            segments = [crack_list[idx] for crack_list in crack_lists if len(crack_list[idx]) > 0]
            joined.append(np.vstack(segments) if segments else np.empty((0, 2, 2)))
        return joined

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_plus_minus(
        cls,
        *,
        name: str,
        angle_deg: float,
        transverse_layer: bool = False,
        scale_px_mm: float,
        path_full: str,
        sorting_key: str,
        image_types: List[str],
        path_upper_border: Optional[str] = None,
        path_lower_border: Optional[str] = None,
        path_middle: Optional[str] = None,
        **kwargs: Any,
    ) -> "Specimen":
        """Create a [+θ, -θ] laminate with two plies (optionally add 90°).

        One interface is added automatically between ``+θ`` and ``-θ``. If
        ``transverse_layer`` is ``True``, a second interface is added
        between ``-θ`` and the 90° ply.

        Example
        -------
        >>> Specimen.from_plus_minus(
        ...     name=\"sample\",
        ...     angle_deg=45,
        ...     scale_px_mm=40.0,
        ...     path_full=\"cut_dir\",
        ...     sorting_key=\"_frame\",
        ...     image_types=[\".png\"],
        ... )
        """
        orientations = [angle_deg, -angle_deg]
        if transverse_layer:
            orientations.append(90.0)
        specimen = cls._build_with_orientations(
            name=name,
            orientations=orientations,
            scale_px_mm=scale_px_mm,
            path_full=path_full,
            sorting_key=sorting_key,
            image_types=image_types,
            path_upper_border=path_upper_border,
            path_lower_border=path_lower_border,
            path_middle=path_middle,
            **kwargs,
        )
        specimen._add_consecutive_interfaces(orientations)
        return specimen

    @classmethod
    def from_cross_ply(
        cls,
        *,
        name: str,
        scale_px_mm: float,
        path_full: str,
        sorting_key: str,
        image_types: List[str],
        angles: Optional[List[float]] = None,
        path_upper_border: Optional[str] = None,
        path_lower_border: Optional[str] = None,
        path_middle: Optional[str] = None,
        **kwargs: Any,
    ) -> "Specimen":
        """Create a cross-ply laminate [0, 90].

        When ``angles`` is left at its default ``[0, 90]``, a single
        ``"0/90"`` interface is added automatically between the two plies.
        A custom ``angles`` sequence adds no interfaces; call
        :meth:`add_interface` yourself in that case.

        Example
        -------
        >>> Specimen.from_cross_ply(
        ...     name=\"cp_sample\",
        ...     scale_px_mm=40.0,
        ...     path_full=\"cut_dir\",
        ...     sorting_key=\"_frame\",
        ...     image_types=[\".png\"],
        ... )
        """
        pattern = angles or [0.0, 90.0]
        specimen = cls._build_with_orientations(
            name=name,
            orientations=pattern,
            scale_px_mm=scale_px_mm,
            path_full=path_full,
            sorting_key=sorting_key,
            image_types=image_types,
            path_upper_border=path_upper_border,
            path_lower_border=path_lower_border,
            path_middle=path_middle,
            **kwargs,
        )
        if angles is None:
            specimen._add_consecutive_interfaces(pattern)
        return specimen

    @classmethod
    def _build_with_orientations(
        cls,
        *,
        name: str,
        orientations: Sequence[float],
        scale_px_mm: float,
        path_full: str,
        sorting_key: str,
        image_types: List[str],
        path_upper_border: Optional[str] = None,
        path_lower_border: Optional[str] = None,
        path_middle: Optional[str] = None,
        **kwargs: Any,
    ) -> "Specimen":
        """Build a specimen and populate plies from an orientation sequence."""
        specimen = cls(
            name=name,
            scale_px_mm=scale_px_mm,
            path_full=path_full,
            path_upper_border=path_upper_border,
            path_lower_border=path_lower_border,
            path_middle=path_middle,
            sorting_key=sorting_key,
            image_types=image_types,
            **kwargs,
        )
        for idx, orientation in enumerate(orientations):
            specimen.add_ply(name=f"ply_{idx}", orientation_deg=float(orientation))
        return specimen

    @staticmethod
    def _format_angle(angle_deg: float) -> str:
        """Render an orientation angle for interface naming, e.g. ``0``, ``-45``."""
        return f"{angle_deg:g}"

    def _add_consecutive_interfaces(self, orientations: Sequence[float]) -> None:
        """Add one interface between each pair of consecutive plies.

        Used by the ``from_cross_ply``/``from_plus_minus`` convenience
        constructors, whose ply count and order are known ahead of time, so
        interfaces can be named unambiguously from the orientation pattern.
        """
        for idx in range(len(orientations) - 1):
            name = f"{self._format_angle(orientations[idx])}/{self._format_angle(orientations[idx + 1])}"
            self.add_interface(name=name, upper_ply_index=idx, lower_ply_index=idx + 1)


__all__ = [
    "Color",
    "Interface",
    "Ply",
    "Specimen",
    "DEFAULT_PLY_COLOR",
    "DEFAULT_CRACK_COLOR",
    "DEFAULT_PRIMARY_DELAMINATION_COLOR",
    "DEFAULT_SECONDARY_DELAMINATION_COLOR",
]
