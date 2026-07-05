"""Shift correction and strain evaluation tooling with a Tkinter GUI.

This module offers a production-ready interface around the original research
prototype. The refactor provides:

* Configurable image suffix and extension handling (no longer bound to
  ``"cycles.bmp"``).
* A modernised Tk GUI with explicit status updates, error handling, and
  settings management.
* Programmatic separation between data-handling (``SpecimenVideo`` /
  ``DIC``) and UI (``ShiftCorrectionApp``).

Several scientific routines (digital image correlation, strain evaluation,
point tracking) remain unchanged, but have been wrapped with additional
safeguards so the application fails gracefully when misconfigured.
"""

from __future__ import annotations

import csv
import logging
import os
import platform
import re
import shutil
import time
import tkinter as tk
from dataclasses import dataclass
from enum import Enum, auto
from functools import lru_cache
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageTk
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from scipy import interpolate, ndimage, sparse, spatial
from skimage import feature, io, morphology, registration


RESAMPLE_LANCZOS = (
    Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
)


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# ---------------------------------------------------------------------------
# Constants & defaults
# ---------------------------------------------------------------------------
DEFAULT_SORTING_KEY = "cycles"
DEFAULT_SORTING_MODE = "suffix"
DEFAULT_FILE_TYPES: Tuple[str, ...] = (
    ".bmp",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
)


class ProcessingMode(Enum):
    """Enum representing the desired processing flow."""

    NONE = auto()
    SHIFT = auto()
    STRAIN = auto()
    BOTH = auto()


@dataclass
class ProcessingSettings:
    """Container for processing parameters tweakable via the GUI."""

    step: int = 1
    median_filter: bool = True
    threshold_value: int = 10
    gaussian_filter: int = 1
    max_distance: int = 10
    sorting_key: str = DEFAULT_SORTING_KEY
    sorting_mode: str = DEFAULT_SORTING_MODE
    file_types: Tuple[str, ...] = DEFAULT_FILE_TYPES


class SpecimenVideo:
    """Lightweight iterator over specimen frames on disk."""

    def __init__(
        self,
        directory: Union[Path, str],
        *,
        sorting_key: Optional[str] = None,
        sorting_mode: str = DEFAULT_SORTING_MODE,
        file_types: Optional[Sequence[str]] = None,
    ) -> None:
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise FileNotFoundError(f"Directory not found: {self.directory}")

        self.sorting_key = (sorting_key or DEFAULT_SORTING_KEY).strip()
        mode_normalised = (sorting_mode or DEFAULT_SORTING_MODE).strip().lower()
        if mode_normalised not in {"prefix", "suffix", "none"}:
            raise ValueError("sorting_mode must be 'prefix', 'suffix', or 'none'")
        self.sorting_mode = mode_normalised

        parsed_types: List[str] = []
        if file_types:
            for extension in file_types:
                ext = extension.strip().lower()
                if not ext:
                    continue
                if not ext.startswith('.'):
                    ext = f'.{ext}'
                parsed_types.append(ext)
        if not parsed_types:
            parsed_types = list(DEFAULT_FILE_TYPES)
        self.file_types = tuple(dict.fromkeys(parsed_types))

        self.image_paths = self._collect_paths()
        if not self.image_paths:
            types_display = ", ".join(self.file_types)
            raise FileNotFoundError(
                f"No images matching extensions {types_display} found in {self.directory}."
            )
        self._validate_uniform_dimensions()

    # ------------------------------------------------------------------
    def _collect_paths(self) -> List[Path]:
        extensions = self.file_types
        sorting_key = self.sorting_key.lower()

        def is_processing_artifact(path: Path) -> bool:
            stem = path.stem.lower()
            return stem.startswith("tracking_") or stem.endswith("_dic_test")

        paths: List[Path] = []
        extension_only_paths: List[Path] = []
        for path in self.directory.iterdir():
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            if is_processing_artifact(path):
                continue
            stem_lower = path.stem.lower()
            extension_only_paths.append(path)
            if sorting_key:
                if self.sorting_mode == "suffix" and not stem_lower.endswith(sorting_key):
                    continue
                if self.sorting_mode == "prefix" and not stem_lower.startswith(sorting_key):
                    continue
            paths.append(path)

        if sorting_key and self.sorting_mode in {"suffix", "prefix"} and not paths:
            LOGGER.warning(
                "No files matched sorting key '%s' (%s mode) in %s; falling back to extension-only matching.",
                self.sorting_key,
                self.sorting_mode,
                self.directory,
            )
            paths = list(extension_only_paths)

        def sort_key(path: Path) -> Tuple[int, str]:
            base_name = path.stem
            candidate = base_name
            if sorting_key:
                if self.sorting_mode == "suffix" and base_name.lower().endswith(sorting_key):
                    candidate = base_name[:-len(sorting_key)] or base_name
                elif self.sorting_mode == "prefix" and base_name.lower().startswith(sorting_key):
                    candidate = base_name[len(sorting_key):] or base_name
            digits = re.findall(r"\d+", candidate)
            numeric = int(digits[-1]) if digits else float("inf")
            return numeric, path.name.lower()

        return sorted(paths, key=sort_key)

    # ------------------------------------------------------------------
    def _validate_uniform_dimensions(self) -> None:
        reference_size: Optional[Tuple[int, int]] = None
        reference_name: Optional[str] = None
        mismatches: List[Tuple[str, Tuple[int, int]]] = []

        for path in self.image_paths:
            try:
                with Image.open(path) as image:
                    size = (int(image.height), int(image.width))
            except Exception as exc:  # noqa: BLE001 - provide user-facing context
                raise ValueError(f"Could not read image '{path.name}' while validating frame sizes: {exc}") from exc

            if reference_size is None:
                reference_size = size
                reference_name = path.name
                continue

            if size != reference_size:
                mismatches.append((path.name, size))
                if len(mismatches) >= 5:
                    break

        if not mismatches:
            return

        expected = f"{reference_size[0]}x{reference_size[1]}" if reference_size else "unknown"
        mismatch_preview = ", ".join(
            f"{name}:{dims[0]}x{dims[1]}" for name, dims in mismatches
        )
        raise ValueError(
            "Images in the selected folder do not share one consistent size. "
            f"Reference {reference_name or '<unknown>'} has {expected}. "
            f"Mismatches: {mismatch_preview}. "
            "Use a clean input folder (only raw frames) or adjust sorting/file types."
        )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.image_paths)

    # ------------------------------------------------------------------
    @lru_cache(maxsize=8)
    def get_image(self, frame: int) -> np.ndarray:
        path = self.image_paths[frame]
        return io.imread(fname=str(path), as_gray=True)


class DIC:
    MAX_DISTANCE = 10   # Max distance in px from the detected point and previous

    def __init__(self, initial_points, median_filter_flag = True, treshold_value = 10, 
                 gaussian_filter_value = 1, max_distance = None, specimen_name = 'default', 
                 output_path = None, image_path = None):
        """
        Initializer for DIC class
        
        Parameters
        ----------
        initial_points : list of tuples
            List of points in the image where the DIC algorithm should start looking for the pattern.
        median_filter_flag : bool, optional
            Whether to use a median filter to pre-process the images. Defaults to True.
        treshold_value : int, optional
            The value below which the pixels are considered to be part of the pattern. Defaults to 10.
        gaussian_filter_value : int, optional
            The sigma value of the gaussian filter to use to pre-process the images. Defaults to 1.
        max_distance : int, optional
            Maximum allowed point-to-point matching distance in pixels. Defaults to ``MAX_DISTANCE``.
        specimen_name : str, optional
            The name of the specimen. Defaults to 'default'.
        output_path : str, optional
            The path where the output files will be saved. Defaults to the current working directory.
        image_path : str, optional
            The path where the images are stored. Defaults to None.
        """
        self.image_slice = (slice(1, -1), slice(1, -1))
        self.list_images = list()
        self.list_points = list()
        self.list_interpolator = list()
        
        # Inputs
        self.median_filter_flag = median_filter_flag
        self.treshold_value = treshold_value
        self.gaussian_filter_value = gaussian_filter_value
        self.max_distance = self.MAX_DISTANCE if max_distance is None else max(1, int(max_distance))
        self.initial_points = initial_points
        self.specimen_name = specimen_name
        
        
        file_path = os.path.dirname(os.path.abspath(__file__))
        self.output_path = output_path if output_path is not None else file_path
        self.image_path = image_path

    def extract_points(self, image): 
        """
        Extracts points from an image that are considered local minima 
        based on filtering and threshold criteria.

        This function processes the input image using a series of filters 
        to enhance features, then identifies local minima points that fall 
        below a specified threshold value. These points are returned as coordinates.

        Parameters
        ----------
        image : array-like
            The input image from which points are extracted.

        Returns
        -------
        points : ndarray
            An array of coordinates of the extracted points.
        """

        image_0 = np.asarray(image, dtype=float)
        
        if self.median_filter_flag:
            image_1 = ndimage.median_filter(image_0, footprint=morphology.disk(5), mode="nearest")
        else:
            image_1 = ndimage.maximum_filter(image_0, footprint=morphology.disk(5), mode="nearest")
            image_1 = ndimage.minimum_filter(image_1, footprint=morphology.disk(5), mode="nearest")
        image_2 = ndimage.gaussian_filter(image_1, sigma=self.gaussian_filter_value, truncate=10, mode="nearest")
        mask = morphology.local_minima(image_2, allow_borders=False)
        mask_2 = np.logical_and(mask, image_2 < self.treshold_value)
        points = np.argwhere(mask_2).astype(float)

        return points

    def match_points(self, ref_points, points):
        """
        Matches points between two sets using bipartite graph matching.

        This function takes two sets of points, `ref_points` and `points`, and finds
        the best matching between them based on spatial proximity. It constructs
        k-D trees for both sets of points and computes a sparse distance matrix 
        within a maximum distance threshold. The distance matrix is then converted
        to a score matrix, and a maximum bipartite matching is performed to find
        the optimal correspondence between the points.

        Parameters
        ----------
        ref_points : ndarray
            An array of reference points to which the other points will be matched.
        points : ndarray
            An array of points that need to be matched to the reference points.

        Returns
        -------
        new_points : ndarray
            An array of points matched to the reference points. Points without a 
            match are filled with NaN values.
        """

        ref_tree = spatial.cKDTree(ref_points)
        tree = spatial.cKDTree(points)

        score_mat = sparse.csr_matrix(
            ref_tree.sparse_distance_matrix(tree, max_distance=self.max_distance, p=2)
        )
        score_mat.data = 1. / (1. + score_mat.data)

        permutation = sparse.csgraph.maximum_bipartite_matching(score_mat, perm_type="column")
        permutation_mask = permutation > -1

        new_points = np.full_like(ref_points, fill_value=float("nan"))
        new_points[permutation_mask] = points[permutation[permutation_mask]]

        return new_points

    def interpolator(self, ref_points, points):
        """
        Creates an RBF interpolator to map reference points to target points.

        This method constructs a Radial Basis Function (RBF) interpolator using
        thin plate spline kernel to interpolate between a set of reference points
        and a corresponding set of target points. It applies a mask to filter
        out non-finite points before creating the interpolator.
        
        Other interpolators can be substituted here as long as this function returns
        a interpolator object.
        
        Parameters
        ----------
        ref_points : ndarray
            An array of reference points used for interpolation.
        points : ndarray
            An array of target points corresponding to the reference points.
        
        Returns
        -------
        RBFInterpolator
            An RBF interpolator object that can be used to interpolate
            values at new points in the same space.
        """

        mask = np.all(np.isfinite(points), axis=1)
        return interpolate.RBFInterpolator(ref_points[mask], points[mask], smoothing=1000,
                                           kernel='thin_plate_spline', degree=1)

    def start(self, image: np.ndarray, plot: bool = False, initial_image_path: Optional[Path] = None, copy_reference: bool = False):
        """
        Starts the shift correction algorithm.

        This method takes an initial image and applies the shift correction algorithm to it.
        It also stores the initial points in a list that will be used to store all the points
        found in each image.

        Parameters
        ----------
        image : ndarray
            The initial image to process.
        plot : bool
            Whether to plot the tracking points or not. Defaults to False.
        initial_image_path : str
            The path of the first image. Defaults to None.

        Returns
        -------
        None
        """
        image = image[self.image_slice]
        self.list_images = [image]
        self.list_points = [self.initial_points]
        #self.list_points = [self.extract_points(image)]
        self.list_interpolator = [self.interpolator(self.list_points[0], self.list_points[-1])]

        # If requested, copy the provided initial image into the shift_corrected
        # output folder as the first shifted image 
        if copy_reference and initial_image_path is not None:
            try:
                save_dir = Path(self.output_path) / self.specimen_name / "shift_corrected"
                save_dir.mkdir(parents=True, exist_ok=True)
                destination = save_dir / "0001_sc.bmp"
                shutil.copy2(str(initial_image_path), str(destination))
                LOGGER.debug("Copied reference image to %s", destination)
            except (OSError, shutil.Error):
                LOGGER.exception("Failed to copy reference image to shift_corrected folder")

        if plot:
            fig, ax = plt.subplots()
            fig.set_size_inches(8, 3)
            ax.set_aspect("equal")
            ax.set_axis_off()
            ax.imshow(np.copy(image), cmap="Greys_r", zorder=1)

            all_points = np.asarray(self.list_points)
            for i in range(all_points.shape[1]):
                ax.plot(all_points[:, i, 1], all_points[:, i, 0], color="b", zorder=9)

            ax.scatter(self.list_points[-1][:, 1], self.list_points[-1][:, 0], color="r", marker=".",
                       s=1, zorder=10)

            save_dir = Path(self.output_path) / self.specimen_name / "tracking"
            save_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(save_dir / "0000_dic_test.png"), dpi=300)
            plt.close(fig)

    def next_image(self, image: np.ndarray):
        image = image[self.image_slice]
        current_points = self.extract_points(image)

        interpolated_points_0 = self.interpolator(
            ref_points=self.list_points[0], points=self.list_points[-1]
        )(self.list_points[0])

        current_points_matched = self.match_points(interpolated_points_0, current_points)

        if len(self.list_images) > 2:
            # Set all previous images to None except the first one
            # self.list_images[-2] = None

            self.list_images[1:-1] = [None] * (len(self.list_images) - 2)
        
        self.list_images.append(image)
        self.list_points.append(current_points_matched)
        self.list_interpolator.append(None)

    def fit_matched_points_accurately(self, idx: int, d: float = 20):
        """
        Refines the points in the image at index `idx` by performing a more accurate
        registration of the pattern around each point.

        Parameters
        ----------
        idx : int
            The index of the image in the sequence to refine points for.
        d : float, optional
            The size of the search window for the registration. Defaults to 20.

        Notes
        -----
        The function takes the points in the image at index `idx` and refines them by
        performing a phase cross correlation registration of the pattern around each point
        in the first image and the pattern in the image at index `idx` around the same point.
        The search window is centered around the point and has a size of `d`.
        """
        
        d = int(d)
        image_0 = self.list_images[0]
        image_idx = self.list_images[idx]
        # Pad both images so points near the border always yield a full 2d×2d patch.
        image_0_ = np.zeros(np.array(image_0.shape) + np.array([2*d, 2*d]), dtype=image_0.dtype)
        image_0_[d:-d, d:-d] = image_0
        image_idx_ = np.zeros(np.array(image_idx.shape) + np.array([2*d, 2*d]), dtype=image_idx.dtype)
        image_idx_[d:-d, d:-d] = image_idx

        points_0 = self.list_points[0]
        points_idx = np.copy(self.list_points[idx])

        mask = np.all(np.isfinite(points_idx), axis=1)
        for i in range(len(points_idx)):
            if mask[i]:
                point_0 = points_0[i].astype(int)
                point_idx = points_idx[i].astype(int)

                shift, error, *_ = registration.phase_cross_correlation(
                    image_0_[point_0[0]:point_0[0] + 2*d, point_0[1]:point_0[1] + 2*d],
                    image_idx_[point_idx[0]:point_idx[0] + 2*d, point_idx[1]:point_idx[1] + 2*d],
                    upsample_factor=5
                )
                points_idx[i] = point_idx - shift
        
        self.list_points[idx] = points_idx
        
    def back_transform_image(self, idx: int, plot: bool = False):
        """
        Back-transforms the image at index `idx` to the reference frame defined by the first image.

        Parameters
        ----------
        idx : int
            The index of the image to be back-transformed.
        plot : bool, optional
            Whether to save the back-transformed image as a PNG file. Defaults to False.

        Returns
        -------
        back_transformed_image : numpy.ndarray
            The back-transformed image.
        """
        ref_image = self.list_images[0]
        image_idx = self.list_images[idx]
        points_0 = self.list_points[0]
        points_idx = self.list_points[idx]

        i, j = [np.arange(ref_image.shape[k]) for k in [0, 1]]
        I, J = np.meshgrid(i, j, indexing="ij")
        ij = np.column_stack([I.flatten(), J.flatten()])

        interp_idx = interpolate.RegularGridInterpolator(
            points=(i, j), values=image_idx, bounds_error=False, fill_value=0
        )
        interp_0_to_idx = self.interpolator(points_0, points_idx)

        back_transformed_image = np.zeros_like(ref_image)
        ij_1 = interp_0_to_idx(ij)
        mask = np.logical_and.reduce(np.isfinite(ij_1), axis=1)
        ij_1[~mask] = 0

        v_idx = interp_idx(ij_1)
        back_transformed_image[ij[:, 0], ij[:, 1]] = v_idx.flatten()

        if plot:
            save_dir = os.path.join(self.output_path, self.specimen_name, "shift_corrected")
            os.makedirs(save_dir, exist_ok=True)
            io.imsave(f"{save_dir}/{len(self.list_points):04d}_sc.png",
                      back_transformed_image)

        return back_transformed_image

    def plot_marker(self, idx: int):
        """
        Plots the tracking of points on the image at the specified index.

        This function generates a plot of the points tracked in the image
        sequence, highlighting the points in the current image at the given index
        with red markers, and drawing lines connecting their positions across all
        images in blue. The plot is saved as a PNG file in the tracking directory.

        Parameters
        ----------
        idx : int
            The index of the image in the sequence for which to plot and save the
            marker tracking visualization.
        """

        image = self.list_images[idx]

        fig, ax = plt.subplots()
        fig.set_size_inches(8, 3)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.imshow(image, cmap="Greys_r", zorder=1)
        all_points = np.asarray(self.list_points)
        for i in range(all_points.shape[1]):
            ax.plot(all_points[:, i, 1], all_points[:, i, 0], color="b", zorder=9)

        ax.scatter(self.list_points[idx][:, 1], self.list_points[idx][:, 0], color="r", marker=".",
                   s=1, zorder=10)
        
        save_dir = os.path.join(self.output_path, self.specimen_name, "tracking")
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(f"{save_dir}/{idx:04d}_dic_test.png", dpi=300)
        plt.close(fig)
        
    def eval_strain(self, idx: int):
        """
        Evaluates the strain between the first and the current image.

        This method uses the points tracked in the image sequence to calculate the
        strain between the first and the current image. The strain is calculated as
        the change in length (in px) between the two images divided by the initial length.

        WARNING: This function is very simplistic and only supports the point layouts
                 used by the current GUI workflows. The results obtained from this 
                 function should be adapted to the specific use case and compared with 
                 other strain measuring methods.
        
        Supported point orders are counter-clockwise:
            4 points:
                1. Top left
                2. Bottom left
                3. Bottom right
                4. Top right
            8 points:
                1. Top left
                2. Middle left
                3. Bottom left
                4. Bottom middle
                5. Bottom right
                6. Middle right
                7. Top right
                8. Top middle
        
        Parameters
        ----------
        idx : int
            The index of the image in the sequence for which to calculate the
            strain.

        Returns
        -------
        strain_x : float
            The transverse strain (x-direction) between the first and the current
            image.
        strain_y : float
            The longitudinal strain (y-direction) between the first and the current
            image.
            
        Notes:
        -------
            x -> is transverse
            y -> is longitudinal
        """
        if idx <= 1:
            strain_x = 0
            strain_y = 0
        else:
            initial_points = self.list_points[1]
            points_idx = self.list_points[idx]

            point_count = len(initial_points)
            if point_count != len(points_idx):
                raise ValueError("Tracked point count changed during processing; cannot evaluate strain.")

            if point_count == 4:
                initial_len_y = [
                    np.abs(initial_points[0][1] - initial_points[3][1]),
                    np.abs(initial_points[1][1] - initial_points[2][1]),
                ]
                current_len_y = [
                    np.abs(points_idx[0][1] - points_idx[3][1]),
                    np.abs(points_idx[1][1] - points_idx[2][1]),
                ]
                initial_len_x = [
                    np.abs(initial_points[0][0] - initial_points[1][0]),
                    np.abs(initial_points[2][0] - initial_points[3][0]),
                ]
                current_len_x = [
                    np.abs(points_idx[0][0] - points_idx[1][0]),
                    np.abs(points_idx[2][0] - points_idx[3][0]),
                ]
            elif point_count == 8:
                initial_len_y = [np.abs(initial_points[0][1] - initial_points[-1][1]),
                                np.abs(initial_points[2][1] - initial_points[3][1]),
                                np.abs(initial_points[3][1] - initial_points[4][1]),
                                np.abs(initial_points[6][1] - initial_points[7][1])]

                current_len_y = [np.abs(points_idx[0][1] - points_idx[-1][1]),
                                np.abs(points_idx[2][1] - points_idx[3][1]),
                                np.abs(points_idx[3][1] - points_idx[4][1]),
                                np.abs(points_idx[6][1] - points_idx[7][1])]

                initial_len_x = [np.abs(initial_points[0][0] - initial_points[1][0]),
                                 np.abs(initial_points[2][0] - initial_points[1][0]),
                                 np.abs(initial_points[4][0] - initial_points[5][0]),
                                 np.abs(initial_points[5][0] - initial_points[6][0])]


                current_len_x = [np.abs(points_idx[0][0] - points_idx[1][0]),
                                 np.abs(points_idx[2][0] - points_idx[1][0]),
                                 np.abs(points_idx[4][0] - points_idx[5][0]),
                                 np.abs(points_idx[5][0] - points_idx[6][0])]
            else:
                raise ValueError(
                    f"Unsupported reference point layout: expected 4 or 8 points, got {point_count}."
                )

            # Filter out lengths below 30 or above 2000
            filtered_initial_len_y = []
            filtered_current_len_y = []
            for ily, cly in zip(initial_len_y, current_len_y):
                if 30 <= ily <= 2000 and 30 <= cly <= 2000:
                    filtered_initial_len_y.append(ily)
                    filtered_current_len_y.append(cly)


            if len(filtered_initial_len_y) > 0 and len(filtered_current_len_y) > 0:
                strains_y = (np.array(filtered_current_len_y) - np.array(filtered_initial_len_y)) / np.array(filtered_initial_len_y)
                strain_y = np.average(strains_y)
            else:
                strain_y = 0

            filtered_initial_len_x = []
            filtered_current_len_x = []

            for  ilx, clx in zip(initial_len_x, current_len_x):
                
                if 30 <= ilx <= 1000 and 30 <= clx <= 1000:
                    filtered_initial_len_x.append(ilx)
                    filtered_current_len_x.append(clx)


            if len(filtered_initial_len_x) > 0 and len(filtered_current_len_x) > 0:
                strains_x = (np.array(filtered_current_len_x) - np.array(filtered_initial_len_x)) / np.array(filtered_initial_len_x)
                strain_x = np.average(strains_x)
            else:
                strain_x = 0


        return strain_x, strain_y


class ShiftCorrectionApp:
    """Tkinter front-end wiring the workflows together."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("DelaDect Shift Correction")
        self.root.geometry("1200x800")
        if platform.system() == "Windows":
            self.root.iconbitmap(default="")  # Safe no-op when icon missing

        self.settings = ProcessingSettings()
        self.processing_mode = ProcessingMode.NONE
        self.processing_in_progress = False

        self.points: List[Tuple[int, int]] = []
        self.photo_image: Optional[ImageTk.PhotoImage] = None
        self.current_image: Optional[Image.Image] = None
        self.initial_image_path: Optional[Path] = None
        self.saved_path: Optional[Path] = None
        self.output_folder: Optional[Path] = None
        self.specimen_video: Optional[SpecimenVideo] = None

        self.zoom_level = 1.0
        self.image_offset = [0, 0]

        self.directory_var = tk.StringVar(value="No directory selected")
        self.sorting_key_var = tk.StringVar(value=self.settings.sorting_key)
        self.sorting_mode_var = tk.StringVar(value=self.settings.sorting_mode)
        self.file_types_var = tk.StringVar(value=", ".join(self.settings.file_types))
        self.status_var = tk.StringVar(value="Select a directory and reference image to begin.")

        self._build_gui()
        self._bind_canvas_events()

    # ------------------------------------------------------------------
    def _build_gui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.columnconfigure(0, weight=4)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # Canvas for image display
        self.canvas = tk.Canvas(main_frame, background="black")
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        # Control panel
        control = ttk.Frame(main_frame)
        control.grid(row=0, column=1, sticky="ns")
        control.columnconfigure(0, weight=1)

        ttk.Label(control, textvariable=self.directory_var, wraplength=260).grid(row=0, column=0, sticky="ew", pady=(0, 10))

        ttk.Label(control, text="Sorting key:").grid(row=1, column=0, sticky="w")
        ttk.Entry(control, textvariable=self.sorting_key_var).grid(row=2, column=0, sticky="ew", pady=(0, 5))

        ttk.Label(control, text="Sorting mode:").grid(row=3, column=0, sticky="w")
        mode_combo = ttk.Combobox(control, textvariable=self.sorting_mode_var, values=("suffix", "prefix", "none"), state="readonly")
        mode_combo.grid(row=4, column=0, sticky="ew", pady=(0, 5))

        ttk.Label(control, text="File types (comma-separated):").grid(row=5, column=0, sticky="w")
        ttk.Entry(control, textvariable=self.file_types_var).grid(row=6, column=0, sticky="ew", pady=(0, 10))

        ttk.Separator(control).grid(row=7, column=0, sticky="ew", pady=10)

        ttk.Button(control, text="Open Image Directory", command=self.open_directory).grid(row=8, column=0, sticky="ew")
        ttk.Button(control, text="Open Reference Image", command=self.open_image).grid(row=9, column=0, sticky="ew", pady=(5, 0))
        ttk.Button(control, text="Set Output Folder", command=self.save_image_in).grid(row=10, column=0, sticky="ew", pady=(5, 0))

        ttk.Separator(control).grid(row=11, column=0, sticky="ew", pady=10)

        ttk.Button(control, text="Settings", command=self.open_settings).grid(row=12, column=0, sticky="ew")
        ttk.Button(control, text="Clear Points", command=self.clear_points).grid(row=13, column=0, sticky="ew", pady=(5, 0))

        ttk.Separator(control).grid(row=14, column=0, sticky="ew", pady=10)

        ttk.Button(control, text="Run Shift Correction", command=lambda: self._request_processing(ProcessingMode.SHIFT)).grid(row=15, column=0, sticky="ew")
        ttk.Button(control, text="Run Strain Evaluation", command=lambda: self._request_processing(ProcessingMode.STRAIN)).grid(row=16, column=0, sticky="ew", pady=(5, 0))
        ttk.Button(control, text="Run Both", command=lambda: self._request_processing(ProcessingMode.BOTH)).grid(row=17, column=0, sticky="ew", pady=(5, 0))

        ttk.Separator(control).grid(row=18, column=0, sticky="ew", pady=10)

        ttk.Button(control, text="Quit", command=self.root.destroy).grid(row=19, column=0, sticky="ew")

        status_frame = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        status_frame.grid(row=1, column=0, sticky="ew")
        ttk.Label(status_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w").pack(fill="x")

    # ------------------------------------------------------------------
    def _bind_canvas_events(self) -> None:
        # Bind left button to start pan; add point only on Ctrl+Click (or Command on macOS)
        self.canvas.bind("<ButtonPress-1>", self._on_button_press)
        self.canvas.bind("<B1-Motion>", self._on_button_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_button_release)
        self.canvas.bind("<MouseWheel>", self.zoom_image)
        if platform.system() == "Darwin":  # macOS event name
            self.canvas.bind("<Button-4>", lambda event: self.zoom_image(event, zoom_in=True))
            self.canvas.bind("<Button-5>", lambda event: self.zoom_image(event, zoom_in=False))

    # ------------------------------------------------------------------
    def update_status(self, message: str) -> None:
        LOGGER.info(message)
        self.status_var.set(message)

    # ------------------------------------------------------------------
    def open_directory(self) -> None:
        directory = filedialog.askdirectory(title="Select image folder")
        if not directory:
            return
        self.saved_path = Path(directory)
        self.directory_var.set(str(self.saved_path))
        self.update_status(f"Selected directory: {self.saved_path}")
        self._prepare_specimen_video()

    # ------------------------------------------------------------------
    def open_image(self) -> None:
        initialdir = str(self.saved_path) if self.saved_path else os.getcwd()
        filetypes = [("Image files", "*.bmp *.png *.jpg *.jpeg *.tif *.tiff"), ("All files", "*.*")]
        file_path = filedialog.askopenfilename(title="Select reference image", initialdir=initialdir, filetypes=filetypes)
        if not file_path:
            return

        self.initial_image_path = Path(file_path)
        self.saved_path = self.initial_image_path.parent
        self.directory_var.set(str(self.saved_path))
        self.update_status(f"Loaded reference image: {self.initial_image_path.name}")

        with Image.open(self.initial_image_path) as img:
            self.current_image = img.convert("L")
        self.display_image()
        self._prepare_specimen_video()

    # ------------------------------------------------------------------
    def save_image_in(self) -> None:
        directory = filedialog.askdirectory(title="Select output folder")
        if not directory:
            return
        self.output_folder = Path(directory)
        self.update_status(f"Output directory set to: {self.output_folder}")

    # ------------------------------------------------------------------
    def _prepare_specimen_video(self) -> None:
        if not self.saved_path:
            return
        try:
            sorting_key = self.sorting_key_var.get().strip()
            sorting_mode = (self.sorting_mode_var.get().strip() or DEFAULT_SORTING_MODE).lower()
            if sorting_mode not in {"suffix", "prefix", "none"}:
                sorting_mode = DEFAULT_SORTING_MODE
                self.sorting_mode_var.set(sorting_mode)

            raw_types = [segment.strip().lower() for segment in re.split(r"[;,\s]+", self.file_types_var.get()) if segment.strip()]
            file_types: Tuple[str, ...]
            if raw_types:
                normalized = []
                for ext in raw_types:
                    if not ext.startswith('.'):
                        ext = f'.{ext}'
                    normalized.append(ext)
                file_types = tuple(dict.fromkeys(normalized))
            else:
                file_types = DEFAULT_FILE_TYPES
                self.file_types_var.set(", ".join(file_types))

            self.settings.sorting_key = sorting_key or DEFAULT_SORTING_KEY
            self.settings.sorting_mode = sorting_mode
            self.settings.file_types = file_types

            self.sorting_key_var.set(self.settings.sorting_key)
            self.sorting_mode_var.set(self.settings.sorting_mode)
            self.file_types_var.set(', '.join(self.settings.file_types))

            self.specimen_video = SpecimenVideo(
                self.saved_path,
                sorting_key=self.settings.sorting_key,
                sorting_mode=self.settings.sorting_mode,
                file_types=self.settings.file_types,
            )
            types_display = ', '.join(self.settings.file_types)
            self.update_status(
                f"Found {len(self.specimen_video)} images using {self.settings.sorting_mode} sorting and extensions {types_display}."
            )
        except FileNotFoundError as exc:
            self.specimen_video = None
            messagebox.showerror("Images not found", str(exc))
            self.update_status("Failed to initialise image sequence; adjust sorting or file types.")

    # ------------------------------------------------------------------
    def add_point(self, event: tk.Event) -> None:
        if not self.current_image:
            return
        x = int((event.x - self.image_offset[0]) / self.zoom_level)
        y = int((event.y - self.image_offset[1]) / self.zoom_level)
        if x < 0 or y < 0 or x >= self.current_image.width or y >= self.current_image.height:
            return
        self.points.append((y, x))  # store as (row, col)
        self.redraw_points()
        self.update_status(f"Point added at (row={y}, col={x}). Total points: {len(self.points)}")

    # ------------------------------------------------------------------
    def _on_button_press(self, event: tk.Event) -> None:
        # Record start of potential pan; actual behavior depends on modifier keys
        self._pan_start = (event.x, event.y)
        # On macOS, Command is the modifier for adding points; on others use Control
        is_add = False
        if platform.system() == "Darwin":
            is_add = bool(event.state & 0x4)  # Command key mapping can vary; keep heuristic
        else:
            is_add = bool(event.state & 0x4)  # Control modifier in Tk reports as bit 0x4 on Windows

        if is_add:
            # Treat this press as an add-point action instead of starting a pan
            self.add_point(event)
            # Clear pan_start so dragging doesn't move the image immediately after adding
            self._pan_start = None

    # ------------------------------------------------------------------
    def _on_button_drag(self, event: tk.Event) -> None:
        # If pan was started, perform panning
        if not hasattr(self, "_pan_start") or self._pan_start is None:
            return
        dx = event.x - self._pan_start[0]
        dy = event.y - self._pan_start[1]
        self._pan_start = (event.x, event.y)
        self.image_offset[0] += dx
        self.image_offset[1] += dy
        self.display_image()

    # ------------------------------------------------------------------
    def _on_button_release(self, event: tk.Event) -> None:
        # End pan gesture
        self._pan_start = None

    # ------------------------------------------------------------------
    def pan_image(self, event: tk.Event) -> None:
        if not hasattr(self, "_last_pan"):
            self._last_pan = (event.x, event.y)
            return
        dx = event.x - self._last_pan[0]
        dy = event.y - self._last_pan[1]
        self._last_pan = (event.x, event.y)
        self.image_offset[0] += dx
        self.image_offset[1] += dy
        self.display_image()

    # ------------------------------------------------------------------
    def zoom_image(self, event: tk.Event, zoom_in: Optional[bool] = None) -> None:
        if zoom_in is None:
            zoom_in = event.delta > 0
        self.zoom_level *= 1.1 if zoom_in else 0.9
        self.zoom_level = max(0.1, min(self.zoom_level, 10.0))
        self.display_image()

    # ------------------------------------------------------------------
    def display_image(self) -> None:
        if not self.current_image:
            return
        width = int(self.current_image.width * self.zoom_level)
        height = int(self.current_image.height * self.zoom_level)
        resized = self.current_image.resize((width, height), RESAMPLE_LANCZOS)
        self.photo_image = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(self.image_offset[0], self.image_offset[1], image=self.photo_image, anchor=tk.NW)
        self.redraw_points()

    # ------------------------------------------------------------------
    def redraw_points(self) -> None:
        self.canvas.delete("points")
        for row, col in self.points:
            display_x = int(col * self.zoom_level) + self.image_offset[0]
            display_y = int(row * self.zoom_level) + self.image_offset[1]
            self.canvas.create_oval(display_x - 2, display_y - 2, display_x + 2, display_y + 2, fill="red", outline="", tags="points")
            self.canvas.create_text(display_x + 4, display_y, text=f"({row},{col})", anchor=tk.NW, fill="red", tags="points")

    # ------------------------------------------------------------------
    def clear_points(self) -> None:
        self.points.clear()
        self.redraw_points()
        self.update_status("All points cleared.")

    # ------------------------------------------------------------------
    def open_settings(self) -> None:
        def save() -> None:
            try:
                self.settings.step = max(1, int(step_var.get()))
                self.settings.median_filter = bool(median_var.get())
                self.settings.threshold_value = int(threshold_var.get())
                self.settings.gaussian_filter = max(0, int(gaussian_var.get()))
                self.settings.max_distance = max(1, int(max_distance_var.get()))

                sorting_key = sorting_key_var.get().strip()
                sorting_mode = (sorting_mode_var.get().strip() or DEFAULT_SORTING_MODE).lower()
                if sorting_mode not in {"suffix", "prefix", "none"}:
                    raise ValueError("Sorting mode must be 'prefix', 'suffix', or 'none'.")

                raw_types = [segment.strip().lower() for segment in re.split(r"[;,\s]+", file_types_var.get()) if segment.strip()]
                normalized_types: List[str] = []
                for ext in raw_types:
                    if not ext.startswith('.'):
                        ext = f'.{ext}'
                    normalized_types.append(ext)
                if not normalized_types:
                    normalized_types = list(DEFAULT_FILE_TYPES)

                self.settings.sorting_key = sorting_key or DEFAULT_SORTING_KEY
                self.settings.sorting_mode = sorting_mode
                self.settings.file_types = tuple(dict.fromkeys(normalized_types))

                self.sorting_key_var.set(self.settings.sorting_key)
                self.sorting_mode_var.set(self.settings.sorting_mode)
                self.file_types_var.set(", ".join(self.settings.file_types))
            except ValueError as exc:
                messagebox.showerror("Invalid settings", str(exc))
                return
            dialog.destroy()
            self.update_status("Settings updated.")

        dialog = tk.Toplevel(self.root)
        dialog.title("Settings")
        dialog.grab_set()

        ttk.Label(dialog, text="Frame interval (n):").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        step_var = tk.StringVar(value=str(self.settings.step))
        ttk.Entry(dialog, textvariable=step_var, width=10).grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(dialog, text="Apply median filter:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        median_var = tk.BooleanVar(value=self.settings.median_filter)
        ttk.Checkbutton(dialog, variable=median_var).grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(dialog, text="Threshold value:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        threshold_var = tk.StringVar(value=str(self.settings.threshold_value))
        ttk.Entry(dialog, textvariable=threshold_var, width=10).grid(row=2, column=1, padx=5, pady=5)

        ttk.Label(dialog, text="Gaussian filter σ:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        gaussian_var = tk.StringVar(value=str(self.settings.gaussian_filter))
        ttk.Entry(dialog, textvariable=gaussian_var, width=10).grid(row=3, column=1, padx=5, pady=5)

        ttk.Label(dialog, text="Max match distance (px):").grid(row=4, column=0, sticky="w", padx=5, pady=5)
        max_distance_var = tk.StringVar(value=str(self.settings.max_distance))
        ttk.Entry(dialog, textvariable=max_distance_var, width=10).grid(row=4, column=1, padx=5, pady=5)

        ttk.Label(dialog, text='Sorting key:').grid(row=5, column=0, sticky='w', padx=5, pady=5)
        sorting_key_var = tk.StringVar(value=self.settings.sorting_key)
        ttk.Entry(dialog, textvariable=sorting_key_var, width=20).grid(row=5, column=1, padx=5, pady=5)

        ttk.Label(dialog, text='Sorting mode:').grid(row=6, column=0, sticky='w', padx=5, pady=5)
        sorting_mode_var = tk.StringVar(value=self.settings.sorting_mode)
        ttk.Combobox(dialog, textvariable=sorting_mode_var, values=('suffix', 'prefix', 'none'), state='readonly', width=17).grid(row=6, column=1, padx=5, pady=5)

        ttk.Label(dialog, text='File types (comma-separated):').grid(row=7, column=0, sticky='w', padx=5, pady=5)
        file_types_var = tk.StringVar(value=', '.join(self.settings.file_types))
        ttk.Entry(dialog, textvariable=file_types_var, width=30).grid(row=7, column=1, padx=5, pady=5)

        ttk.Button(dialog, text='Save', command=save).grid(row=8, column=0, columnspan=2, pady=10)

    # ------------------------------------------------------------------
    def _request_processing(self, mode: ProcessingMode) -> None:
        if self.processing_in_progress:
            messagebox.showinfo("Processing", "A processing task is already running.")
            return
        if not self.points:
            messagebox.showinfo("Points required", "Select at least one reference point before running.")
            return
        if not self.initial_image_path:
            messagebox.showinfo("Reference image", "Please load a reference image first.")
            return
        self.processing_mode = mode
        self.root.after(50, self.run_processing)

    # ------------------------------------------------------------------
    def run_processing(self) -> None:
        if self.processing_mode == ProcessingMode.NONE:
            return
        if not self.specimen_video:
            self._prepare_specimen_video()
        if not self.specimen_video:
            return

        self.processing_in_progress = True
        try:
            output_dir = self.output_folder or (self.saved_path / "shift_correction_output" if self.saved_path else Path.cwd() / "shift_correction_output")
            output_dir.mkdir(parents=True, exist_ok=True)

            points_array = np.array(self.points, dtype=int)

            modes = {
                ProcessingMode.SHIFT: (True, False),
                ProcessingMode.STRAIN: (False, True),
                ProcessingMode.BOTH: (True, True),
            }
            shift_flag, strain_flag = modes.get(self.processing_mode, (False, False))

            dic = DIC(
                points_array,
                median_filter_flag=self.settings.median_filter,
                treshold_value=self.settings.threshold_value,
                gaussian_filter_value=self.settings.gaussian_filter,
                max_distance=self.settings.max_distance,
                specimen_name=self.specimen_video.directory.name,
                output_path=output_dir,
                image_path=self.saved_path or output_dir,
            )

            self.update_status("Starting processing...")
            LOGGER.info("Processing started: specimen=%s, step=%d", self.specimen_video.directory.name, self.settings.step)

            dic.start(
                self.specimen_video.get_image(0),
                plot=True,
                initial_image_path=self.initial_image_path,
                copy_reference=shift_flag,
            )

            strain_x_list: List[float] = []
            strain_y_list: List[float] = []

            # Build list of indices we'll process so we can report progress
            indices = list(range(1, len(self.specimen_video), self.settings.step))
            total = len(indices)
            start_time = None
            for counter, idx in enumerate(indices, start=1):
                if start_time is None:
                    start_time = time.monotonic()

                image = self.specimen_video.get_image(idx)
                dic.next_image(image)
                dic.fit_matched_points_accurately(len(dic.list_images) - 1)
                if shift_flag:
                    dic.back_transform_image(len(dic.list_images) - 1, plot=True)
                if strain_flag:
                    strain_x, strain_y = dic.eval_strain(len(dic.list_images) - 1)
                    strain_x_list.append(strain_x)
                    strain_y_list.append(strain_y)
                dic.plot_marker(len(dic.list_images) - 1)

                # Progress reporting
                elapsed = time.monotonic() - start_time
                avg = elapsed / counter
                remaining = avg * (total - counter)
                percent = 100.0 * (counter / total) if total else 100.0
                status_msg = f"Processing image {counter}/{total} (frame {idx}) - {percent:.1f}% - ETA {int(remaining)}s"
                self.update_status(status_msg)

            if strain_flag:
                csv_path = output_dir / "strain_data.csv"
                with csv_path.open("w", newline="", encoding="utf-8") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=["strain_x", "strain_y"])
                    writer.writeheader()
                    for x_val, y_val in zip(strain_x_list, strain_y_list):
                        writer.writerow({"strain_x": x_val, "strain_y": y_val})
                self.update_status(f"Strain evaluation complete. Results written to {csv_path}.")

            if shift_flag and not strain_flag:
                self.update_status("Shift correction complete.")
            elif shift_flag and strain_flag:
                self.update_status("Shift correction and strain evaluation complete.")
            elif strain_flag:
                self.update_status("Strain evaluation complete.")

            messagebox.showinfo("Processing finished", self.status_var.get())
        except Exception as exc:  # noqa: BLE001 - broad catch keeps UI responsive
            LOGGER.exception("Processing failed")
            messagebox.showerror("Processing failed", str(exc))
            self.update_status("Processing failed; see logs for details.")
        finally:
            self.processing_in_progress = False
            self.processing_mode = ProcessingMode.NONE

    # ------------------------------------------------------------------
    def mainloop(self) -> None:
        self.root.mainloop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    root = tk.Tk()
    app = ShiftCorrectionApp(root)
    app.mainloop()


if __name__ == "__main__":
    main()
