"""
Delamination detection module for DelaDect.

Analyzes delamination in composite materials using image analysis and crack data.
Supports upper/lower edge regions and diffuse delamination around cracks.

Key Features:
    - Edge delamination in upper & lower border slices
    - Diffuse delamination around cracks in middle slice
    - Otsu auto-threshold (with manual override, offsets, and multipliers)
    - Optional accumulation vs. previous frames
    - Plotting with optional background and crack overlays
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.io import imread

# Imported for type context; no runtime dependency on this symbol’s contents.
from deladect.detection import Specimen


# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------


def _as_bool_mask(mask: NDArray) -> NDArray[np.bool_]:
    """Convert an array to a boolean mask (True where values > 0).

    Args:
        mask: Input array of any numeric dtype.

    Returns:
        Boolean mask where True indicates positive values.
    """
    return np.asarray(mask) > 0


def edge_rows(mask: NDArray[np.uint8], side: str) -> Optional[NDArray[np.float64]]:
    """Calculate the edge row indices for each column in a binary mask.

    For a given binary mask and side ('top' or 'bottom'), returns an array containing
    the row index of the first/last non-zero value in each column. Uses NaN for
    columns without any non-zero values. Returns `None` if input is empty.

    Args:
        mask: Binary image mask where non-zero values indicate detections. Shape (H, W).
        side: Which edge to detect, one of {'top', 'bottom'}.

    Returns:
        Optional[NDArray[np.float64]]: Array of length W with row indices per column.
        NaN values indicate columns with no detections. Returns None if mask has no hits.

    Raises:
        ValueError: If `side` is not 'top' or 'bottom'.
    """
    if mask is None or mask.size == 0 or not np.any(mask):
        return None

    mask_bool = _as_bool_mask(mask)
    h, _ = mask_bool.shape
    row_idx = np.arange(h, dtype=np.int32).reshape(-1, 1)

    if side == "bottom":
        rows = np.where(mask_bool, row_idx, -1).max(axis=0).astype(np.float64)
        rows[rows < 0] = np.nan
    elif side == "top":
        rows = np.where(mask_bool, row_idx, h).min(axis=0).astype(np.float64)
        rows[rows >= h] = np.nan
    else:
        raise ValueError("side must be 'bottom' or 'top'")
    return rows


def plot_edge(
    ax: plt.Axes,
    start_row: Optional[int],
    cols_w: int,
    rows_local: Optional[NDArray[np.float64]],
    **kwargs,
) -> None:
    """Plot an edge line recovered by :func:`edge_rows`.

    Args:
        ax: Matplotlib axes to draw on.
        start_row: Vertical offset (row index) where the slice starts on the canvas.
        cols_w: Number of columns in the slice.
        rows_local: Row indices per column as returned by :func:`edge_rows`. May include NaNs.
        **kwargs: Additional keyword arguments passed to :meth:`matplotlib.axes.Axes.plot`.

    Returns:
        None
    """
    if rows_local is None or start_row is None:
        return
    x = np.arange(cols_w)
    y = start_row + rows_local  # rows_local may include NaNs
    ax.plot(x, y, **kwargs)


def to_binary_mask(arr: Optional[NDArray]) -> Optional[NDArray[np.uint8]]:
    """Convert arbitrary array to a binary uint8 mask (0/1).

    Args:
        arr: Input array or None.

    Returns:
        Optional[NDArray[np.uint8]]: Binary mask with values {0, 1}, or None if input is None.
    """
    if arr is None:
        return None
    return (_as_bool_mask(arr)).astype(np.uint8)


def has_hits(arr: Optional[NDArray]) -> bool:
    """Check if an array contains any positive values.

    Args:
        arr: Input array or None.

    Returns:
        bool: True if array is non-empty and contains at least one positive value.
    """
    return arr is not None and arr.size > 0 and np.any(arr)


# -------------------------------------------------------------------
# Detector
# -------------------------------------------------------------------

class DelaminationDetector:
    """Detect delamination in a specimen.

    The detector analyzes three regions:
      * Upper edge
      * Lower edge
      * Middle (diffuse) region around cracks

    Args:
        specimen: Specimen instance with image paths and metadata about the specimen's properties.
        cracks_diffuse: Cracks to use for diffuse analysis (affects results).
        cracks_original: Cracks to plot for visualization (does not affect results).
        window_size_edge: Window size (rows, cols) for pre-filtering in edge regions.
            If any dimension is <= 0, it is coerced to 1 internally.
        window_size_diffuse: Window size (rows, cols) for pre-filtering diffuse regions.
            If None, defaults to (3×avg_crack_width_px, 3×avg_crack_width_px).
        gaussian_filters: Sigma(s) for Gaussian filter (y_sigma, x_sigma).
        min_pixel_value: Minimum post-smoothing intensity (0..255) to keep a pixel as positive.
        diffuse_dx: Half-width (in pixels) of ROI around each crack in x for diffuse analysis.
        diffuse_dy: Half-height (in pixels) of ROI around each crack in y for diffuse analysis.
        compare_with_previous: If True, accumulates masks across frames when calling `detect()` repeatedly
            or via `detect_all()`.

    Attributes:
        specimen: The provided specimen.
        cracks: Cracks used for diffuse analysis.
        cracks_orig: Cracks used only for visualization.
        compare_with_previous: Whether to accumulate detections across frames.
        window_size_edge: Effective edge pre-filter window (coerced to >= 1).
        window_size_diffuse: Effective diffuse pre-filter window.
        gaussian_filters: Effective Gaussian filter sigmas.
        min_pixel_value: Effective min pixel threshold post-smoothing.
        diffuse_dx: Effective half-width for diffuse ROI.
        diffuse_dy: Effective half-height for diffuse ROI.

    Examples:
        Basic usage:

        >>> specimen = Specimen(...)  # already prepared
        >>> cracks_90, *_ = specimen.crack_eval_crossply()
        >>> avg_crack_width_px = getattr(specimen, "avg_crack_width_px", 1.0)
        >>> detector = DelaminationDetector(
        ...     specimen=specimen,
        ...     cracks_diffuse=[cracks_90, ...],
        ...     cracks_original=[...],  # optional, for plotting
        ...     window_size_edge=(1, 60),
        ...     window_size_diffuse=(3.0*avg_crack_width_px, 3.0*avg_crack_width_px),
        ...     diffuse_dx=2.0*avg_crack_width_px,
        ...     diffuse_dy=0.0,
        ... )
        >>> areas_px = detector.detect(idx=0)  # [upper_px, lower_px, diffuse_px]
        >>> results = detector.detect_all(plot=True, th_upper=145, th_lower=145, th_offset_middle=-25.0)
    """

    def __init__(
        self,
        specimen: Specimen,
        cracks_diffuse: Optional[Any] = None,
        cracks_original: Optional[Any] = None,
        window_size_edge: Tuple[int, int] = (0, 60),
        window_size_diffuse: Optional[Tuple[float, float]] = None,
        gaussian_filters: Tuple[float, float] = (0.5, 15.0),
        min_pixel_value: int = 128,
        diffuse_dx: float = 20.0,
        diffuse_dy: float = 20.0,
        compare_with_previous: bool = True,
    ):
        self.specimen = specimen
        self.cracks: Optional[Any] = cracks_diffuse
        self.cracks_orig: Optional[Any] = cracks_original
        self.compare_with_previous = bool(compare_with_previous)

        self.specimen_name = getattr(specimen, "name", "unknown_specimen")


        # --- store/normalize parameters (no external config class) ---
        wy, wx = int(window_size_edge[0]), int(window_size_edge[1])
        self.window_size_edge: Tuple[int, int] = (max(1, wy), max(1, wx))

        if window_size_diffuse is None:
            avg_crack_width_px = float(getattr(specimen, "avg_crack_width_px", 1.0))
            self.window_size_diffuse: Tuple[int, int] = (
                int(max(1, round(3.0 * avg_crack_width_px))),
                int(max(1, round(3.0 * avg_crack_width_px))),
            )
        else:
            wdy, wdx = float(window_size_diffuse[0]), float(window_size_diffuse[1])
            self.window_size_diffuse = (int(max(1, round(wdy))), int(max(1, round(wdx))))

        self.gaussian_filters: Tuple[float, float] = (float(gaussian_filters[0]), float(gaussian_filters[1]))
        self.min_pixel_value: int = int(max(0, min(255, min_pixel_value)))
        self.diffuse_dx: float = float(diffuse_dx)
        self.diffuse_dy: float = float(diffuse_dy)

        self._prev_upper: Optional[NDArray[np.uint8]] = None
        self._prev_middle: Optional[NDArray[np.uint8]] = None
        self._prev_lower: Optional[NDArray[np.uint8]] = None

    # ------------------------ Core operations ------------------------

    def apply_filters(
        self,
        image: NDArray[np.uint8],
        th: float,
        window_size: Tuple[int, int],
        min_pixel_value: Optional[int] = None,
    ) -> NDArray[np.uint8]:
        """Apply the filtering/thresholding pipeline and return a binary mask.

        The pipeline is:
          1. Maximum filter
          2. Minimum filter
          3. Threshold (binary)
          4. Gaussian smoothing
          5. Vertical minimum accumulate (column-wise)

        Args:
            image: Input grayscale image (uint8). Shape (H, W).
            th: Threshold value to compare against after min filter.
            window_size: Window size (rows, cols) for max/min pre-filtering (coerced to >=1).
            min_pixel_value: Minimum post-smoothing intensity to keep a pixel;
                defaults to `self.min_pixel_value`.

        Returns:
            NDArray[np.uint8]: Binary mask (0/1) of detected regions.
        """
        wy, wx = max(1, int(window_size[0])), max(1, int(window_size[1]))
        filtered_max = ndimage.maximum_filter(image, size=(wy, wx))
        filtered_min = ndimage.minimum_filter(filtered_max, size=(wy, wx))
        binary255 = (filtered_min < th).astype(np.uint8) * 255

        smoothed = ndimage.gaussian_filter(binary255, self.gaussian_filters)
        mpv = self.min_pixel_value if min_pixel_value is None else int(min_pixel_value)
        binary = (smoothed > mpv).astype(np.uint8)

        # Column-wise accumulation (vertical)
        result = np.minimum.accumulate(binary, axis=0)
        return result

    def images_threshold(
        self,
        image: NDArray[np.uint8],
        window_size: Optional[Tuple[int, int]] = None,
    ) -> float:
        """Compute the Otsu threshold on a pre-filtered image.

        Applies maximum and minimum filters before Otsu to improve bimodality.

        Args:
            image: Input grayscale image (uint8). Shape (H, W).
            window_size: Window size (rows, cols) for pre-filtering. If None, uses `self.window_size_edge`.

        Returns:
            float: Otsu threshold value.
        """
        if window_size is None:
            window_size = self.window_size_edge
        wy, wx = max(1, int(window_size[0])), max(1, int(window_size[1]))
        filtered_max = ndimage.maximum_filter(image, size=(wy, wx))
        filtered_min = ndimage.minimum_filter(filtered_max, size=(wy, wx))
        return float(threshold_otsu(filtered_min))

    def area_evaluation(
        self,
        image: NDArray[np.uint8],
        im_type: str = "Upper",
        th: Optional[float] = None,
    ) -> NDArray[np.uint8]:
        """Evaluate edge delamination for 'Upper' or 'Lower' region and return a mask.

        For 'Lower', the input image is flipped vertically for processing to match
        the plotting/orientation logic.

        Args:
            image: Input grayscale image (uint8) of the edge slice.
            im_type: Region type, either 'Upper' or 'Lower'.
            th: Optional manual threshold value. If None, auto-calculated with Otsu.

        Returns:
            NDArray[np.uint8]: Binary mask (0/1) for detected delamination.

        Examples:
            >>> mask = detector.area_evaluation(im_upper, im_type="Upper", th=None)
            >>> np.count_nonzero(mask)
        """
        img = np.flipud(image) if im_type == "Lower" else image
        th_val = self.images_threshold(img) if th is None else float(th)
        mask = self.apply_filters(img, th_val, self.window_size_edge)
        return mask

    # ------------------------ Diffuse Delamination ------------------------
    def diffuse_delamination_image(
        self,
        image: NDArray[np.uint8],
        cracks: Optional[NDArray],
        th: Optional[float] = None,
    ) -> NDArray[np.uint8]:
        """
        Generate delamination mask for diffuse regions around cracks.

        Old-behavior compatible:
        - Pre-filter (max/min) is applied to the *full image*
        - Global Otsu if `th` is None (no diffuse-specific window override)
        - Gaussian smoothing + single threshold (no vertical accumulate)
        - Stitch per-crack ROIs into an output mask using max union

        Args:
            image: Input grayscale image (uint8), shape (H, W)
            cracks: Array of crack endpoints [(y1,x1), (y2,x2)]
            th: Manual threshold; auto-calculated with Otsu if None

        Returns:
            uint8 binary mask (0/1) for diffuse delamination.
        """
        h, w = image.shape[:2]
        output_image = np.zeros((h, w), dtype=np.uint8)

        if not (cracks is not None and getattr(cracks, "size", 0) > 0):
            return output_image

        # Old code based its Otsu on the same call without an explicit diffuse window.
        if th is None:
            th = self.images_threshold(image)

        # --- Apply the old-style pipeline to the full image ---
        # (max -> min -> threshold -> gaussian -> binary @ min_pixel_value)
        filtered_max = ndimage.maximum_filter(image, size=self.window_size_diffuse)
        filtered_min = ndimage.minimum_filter(filtered_max, size=self.window_size_diffuse)
        binary255 = (filtered_min < float(th)).astype(np.uint8) * 255
        smoothed = ndimage.gaussian_filter(binary255, self.gaussian_filters)
        full_processed = (smoothed > 160).astype(np.uint8)  # keep the old 160 default

        # --- Stitch ROIs for each crack (fixes the old slicing bug) ---
        for crack in cracks:
            (y1, x1), (y2, x2) = crack
            # Symmetric padding around BOTH crack endpoints (old style)
            dx = float(self.diffuse_dx)
            dy = float(self.diffuse_dy)
            x_lo = int(max(0, min(x1, x2) - dx))
            x_hi = int(min(w, max(x1, x2) + dx))
            y_lo = int(max(0, min(y1, y2) - dy))
            y_hi = int(min(h, max(y1, y2) + dy))
            if x_hi <= x_lo or y_hi <= y_lo:
                continue

            roi = full_processed[y_lo:y_hi, x_lo:x_hi]
            # Union into the output using max (avoids overwriting prior cracks)
            output_image[y_lo:y_hi, x_lo:x_hi] = np.maximum(
                output_image[y_lo:y_hi, x_lo:x_hi], roi
            )

        return output_image


    def diffuse_delamination_each_crack(
        self,
        image: NDArray[np.uint8],
        crack: NDArray,
        th: float,
        min_pixel_value: int = 160,
    ) -> Tuple[NDArray[np.uint8], int, int, int, int]:
        """
        Old-behavior compatible per-crack processing:
        - ROI bounds padded symmetrically by diffuse_dx/diffuse_dy
        - Filtering is performed on the *full image* (as before),
            then cropped to ROI for return.
        - No vertical accumulate step.

        Returns:
            (roi_mask, start_y, end_y, start_x, end_x)
        """
        h, w = image.shape[:2]
        (y1, x1), (y2, x2) = crack

        # ROI bounds (symmetric around min/max of endpoints)
        dx = float(self.diffuse_dx)
        dy = float(self.diffuse_dy)
        start_x = int(max(0, min(x1, x2) - dx))
        end_x   = int(min(w, max(x1, x2) + dx))
        start_y = int(max(0, min(y1, y2) - dy))
        end_y   = int(min(h, max(y1, y2) + dy))

        if end_x <= start_x or end_y <= start_y:
            return np.zeros((0, 0), dtype=np.uint8), start_y, end_y, start_x, end_x

        # Apply the old pipeline to the full image, then crop
        filtered_max = ndimage.maximum_filter(image, size=self.window_size_diffuse)
        filtered_min = ndimage.minimum_filter(filtered_max, size=self.window_size_diffuse)
        binary255 = (filtered_min < float(th)).astype(np.uint8) * 255
        smoothed = ndimage.gaussian_filter(binary255, self.gaussian_filters)
        result_full = (smoothed > int(min_pixel_value)).astype(np.uint8)

        roi_mask = result_full[start_y:end_y, start_x:end_x]
        return roi_mask, start_y, end_y, start_x, end_x


    # ------------------------ Areas ------------------------

    @staticmethod
    def calculate_area(filtered_image: NDArray[np.uint8]) -> float:
        """Calculate total detected area (in pixels) from a binary mask.

        Args:
            filtered_image: Binary mask (0/1).

        Returns:
            float: Number of non-zero pixels.

        Examples:
            >>> area_px = DelaminationDetector.calculate_area(mask)
            >>> area_px >= 0
            True
        """
        return float(np.count_nonzero(filtered_image))

    # ------------------------ Cracks retrieval ------------------------

    def get_frame_cracks(self, frame_idx: int) -> NDArray:
        """Retrieve all cracks for a specific frame, stacked across orientations.

        Prefers `cracks_original` if available (for plotting), otherwise uses `cracks_diffuse`.

        Args:
            frame_idx: Index of the frame to extract cracks for.

        Returns:
            NDArray: Array with shape (N, 2, 2) (two endpoints per crack).
                        Returns an empty array if no cracks are found.

        Examples:
            >>> c = detector.get_frame_cracks(0)
            >>> c.shape[1:] == (2, 2)
            True
        """
        cracks_data = self.cracks
        if cracks_data is None:
            return np.array([])

        frame_cracks = []
        for direction in cracks_data:
            try:
                crack_list = direction[frame_idx]
                if len(crack_list) > 0:
                    frame_cracks.append(crack_list)
            except (IndexError, TypeError):
                continue

        return np.vstack(frame_cracks) if frame_cracks else np.array([])

    # ------------------------ Plotting ------------------------

    def plot_output(
        self,
        background_image=None,
        show_cracks: bool = True,
        filtered_upper: Optional[NDArray[np.uint8]] = None,
        filtered_lower: Optional[NDArray[np.uint8]] = None,
        filtered_middle: Optional[NDArray[np.uint8]] = None,
        cracks: Optional[NDArray] = None,
        index: int = 0,
        output_dir: str = "output_images",
    ) -> None:
        """Plot and save delamination overlays.

        If `background_image` is provided, masks are placed on the original specimen image.
        Otherwise masks are stacked (upper → middle → lower) on a white canvas.

        Args:
            background_image: Background image to plot under the overlays (uint8, shape (H, W)), or None.
            show_cracks: If True, overlay cracks (when provided).
            filtered_upper: Binary mask (0/1) for upper edge region slice.
            filtered_lower: Binary mask (0/1) for lower edge region slice (will be flipped for display).
            filtered_middle: Binary mask (0/1) for diffuse region slice.
            cracks: Array of cracks to plot (N, 2, 2).
            index: Frame index; used for output filename.
            output_dir: Directory to save images.

        Returns:
            None

        Examples:
            >>> detector.plot_output(
            ...     background_image=None,
            ...     filtered_upper=mask_u, filtered_lower=mask_l, filtered_middle=mask_m,
            ...     cracks=detector.get_frame_cracks(0), index=0,
            ... )
        """
        import os
        from matplotlib.colors import ListedColormap
        import matplotlib.patches as patches

        # Output dir
        full_out_dir = os.path.join(output_dir, self.specimen_name)
        os.makedirs(full_out_dir, exist_ok=True)

        # Heights/widths
        hu = filtered_upper.shape[0] if filtered_upper is not None else 0
        hm = filtered_middle.shape[0] if filtered_middle is not None else 0
        hl = filtered_lower.shape[0] if filtered_lower is not None else 0
        widths = [arr.shape[1] for arr in (filtered_upper, filtered_middle, filtered_lower) if arr is not None]
        W = max(widths) if widths else 100

        # Canvas size
        if background_image is not None:
            H, W_bg = background_image.shape[0], background_image.shape[1]
            W = W_bg if W_bg is not None else W
        else:
            H = max(hu + hm + hl, 100)

        # Helpers
        def pad_w(arr: Optional[NDArray[np.uint8]], target_w: int) -> Optional[NDArray[np.uint8]]:
            """Pad/crop an array horizontally to target width.

            Args:
                arr: Input 2D array or None.
                target_w: Target width.

            Returns:
                Optional[NDArray[np.uint8]]: Padded/cropped array or None if input is None.
            """
            if arr is None:
                return None
            h, w = arr.shape
            if w == target_w:
                return arr
            if w > target_w:
                return arr[:, :target_w]
            pad_right = target_w - w
            return np.pad(arr, ((0, 0), (0, pad_right)), mode="constant", constant_values=0)

        # Lower slice was computed on flipped input; flip back before placement logic
        if filtered_lower is not None:
            filtered_lower = np.flipud(filtered_lower)

        filtered_upper = pad_w(filtered_upper, W)
        filtered_middle = pad_w(filtered_middle, W)
        filtered_lower = pad_w(filtered_lower, W)

        sum_slices = hu + hm + hl
        if background_image is not None and sum_slices == H and sum_slices > 0:
            start_upper = 0 if hu > 0 else None
            start_middle = (start_upper or 0) + hu if hm > 0 else None
            start_lower = (start_middle or 0) + hm if hl > 0 else None
        else:
            start_upper = 0 if hu > 0 else None
            start_lower = (H - hl) if hl > 0 else None
            if hm > 0:
                if hl > 0:
                    start_middle = (start_lower or H) - hm
                elif start_upper is not None:
                    start_middle = start_upper + hu
                else:
                    start_middle = max((H - hm) // 2, 0)
            else:
                start_middle = None

        def place(h_start: Optional[int], sl: Optional[NDArray[np.uint8]]) -> NDArray[np.uint8]:
            """Place a slice at a given vertical start on an (H, W) canvas.

            Args:
                h_start: Start row (top) for placement; clipped to valid range.
                sl: Slice to place.

            Returns:
                NDArray[np.uint8]: Canvas with slice placed (0 elsewhere).
            """
            out = np.zeros((H, W), dtype=np.uint8)
            if sl is None:
                return out
            s = 0 if h_start is None else int(max(0, min(H - sl.shape[0], h_start)))
            out[s:s + sl.shape[0], :sl.shape[1]] = sl
            return out

        full_upper = place(start_upper, filtered_upper)
        full_middle = place(start_middle, filtered_middle)
        full_lower = place(start_lower, filtered_lower)

        # Figure
        fig, ax = plt.subplots(figsize=(8, 8))
        if background_image is not None:
            ax.imshow(background_image, cmap="Greys_r", vmin=0, vmax=255, origin="upper")
        else:
            ax.imshow(np.full((H, W), 255, dtype=np.uint8), cmap="Greys_r", vmin=0, vmax=255, origin="upper")
            rect = patches.Rectangle((0, 0), W, H, linewidth=2, edgecolor="black", facecolor="none")
            ax.add_patch(rect)

        cmap_upper = ListedColormap([[1, 0, 0, 0], [1, 0, 0, 0.8]])
        cmap_middle = ListedColormap([[0, 1, 0, 0], [0, 0.90, 0, 0.8]])
        cmap_lower = ListedColormap([[1, 0, 0, 0], [1, 0, 0, 0.8]])

        if hu > 0 and background_image is None:
            ax.imshow(full_upper, cmap=cmap_upper, vmin=0, vmax=1, alpha=0.5, origin="upper")
        if hm > 0:
            ax.imshow(full_middle, cmap=cmap_middle, vmin=0, vmax=1, alpha=0.5, origin="upper")
        if hl > 0 and background_image is None:
            ax.imshow(full_lower, cmap=cmap_lower, vmin=0, vmax=1, alpha=0.5, origin="upper")

        ref = background_image if background_image is not None else full_middle
        if show_cracks and cracks is not None and getattr(cracks, "size", 0) > 0:
            # Adjust crack coordinates by adding the height of upper region
            shifted_cracks = cracks.copy()
            if hu > 0:  # If there is an upper region, shift cracks down
                shifted_cracks[..., 0] += hu  # Add upper height to all y-coordinates
            self.specimen.plot_cracks(ref, shifted_cracks, color="black", ax=ax)

        if background_image is not None:
            if hu > 0 and filtered_upper is not None:
                rows_u = edge_rows(filtered_upper, side="bottom")
                plot_edge(ax, start_upper, filtered_upper.shape[1], rows_u, linestyle="-", linewidth=1, color="red")
            if hl > 0 and filtered_lower is not None:
                rows_l = edge_rows(filtered_lower, side="top")
                plot_edge(ax, start_lower, filtered_lower.shape[1], rows_l, linestyle="-", linewidth=1, color="red")

        ax.axis("off")
        out_path = Path(output_dir) / self.specimen_name / f"output_{index}.png"
        fig.savefig(out_path.as_posix(), bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)

    # ------------------------ Top-level detection ------------------------

    def _ensure_prev_buffers(self) -> None:
        """Ensure previous-frame buffers exist when accumulation is enabled.

        Lazily initializes zero arrays that match the first frame sizes for
        upper, middle, and lower slices.

        Returns:
            None
        """
        if not self.compare_with_previous:
            return
        if self._prev_upper is None:
            im0 = imread(self.specimen.spec_upper.image_paths[0])
            self._prev_upper = np.zeros_like(im0, dtype=np.uint8)
        if self._prev_middle is None:
            im0 = imread(self.specimen.spec_middle.image_paths[0])
            self._prev_middle = np.zeros_like(im0, dtype=np.uint8)
        if self._prev_lower is None:
            im0 = imread(self.specimen.spec_lower.image_paths[0])
            self._prev_lower = np.zeros_like(im0, dtype=np.uint8)

    def detect(
        self,
        idx: int = 0,
        plot: bool = False,
        background: bool = False,
        th_upper: Optional[float] = None,
        th_lower: Optional[float] = None,
        th_middle: Optional[float] = None,
        show_cracks: bool = True,
    ) -> List[float]:
        """Run delamination detection for a single frame.

        Args:
            idx: Frame index to analyze.
            plot: Whether to plot and save overlays for this frame.
            background: If True, include the original specimen background in the plot.
            th_upper: Manual threshold for upper edge (optional).
            th_lower: Manual threshold for lower edge (optional).
            th_middle: Manual threshold for diffuse region (optional).
            show_cracks: If True, overlay cracks on the plot.

        Returns:
            List[float]: [area_upper_px, area_lower_px, area_diffuse_px], all in pixels.

        Examples:
            >>> upper_px, lower_px, diff_px = detector.detect(
            ...     idx=0, plot=True, background=False, th_upper=145, th_lower=145
            ... )
        """
        self._ensure_prev_buffers()

        # images
        im_original = self.specimen.self.image_stack_cut[idx]
        im_upper = self.specimen.image_stack_upper[idx]
        im_lower = self.specimen.image_stack_lower[idx]
        im_middle = self.specimen.image_stack_middle[idx]



        # image_original = imread(p_orig) if background else None
        # im_upper = imread(p_up)
        # im_lower = imread(p_lo)
        # im_middle = imread(p_mid)

        # cracks
        frame_cracks = self.get_frame_cracks(idx)
        


        # build masks
        mask_upper = self.area_evaluation(im_upper, im_type="Upper", th=th_upper)
        mask_lower = self.area_evaluation(im_lower, im_type="Lower", th=th_lower)
        mask_middle = self.diffuse_delamination_image(im_middle, frame_cracks, th=th_middle)

        if self.compare_with_previous:
            mask_upper = np.clip((self._prev_upper + mask_upper), 0, 1)
            mask_lower = np.clip((self._prev_lower + mask_lower), 0, 1)
            mask_middle = np.clip((self._prev_middle + mask_middle), 0, 1)
            self._prev_upper = mask_upper
            self._prev_lower = mask_lower
            self._prev_middle = mask_middle

        # areas
        area_upper = self.calculate_area(mask_upper)
        area_lower = self.calculate_area(mask_lower)
        area_diffuse = self.calculate_area(mask_middle)


        if plot:
            self.plot_output(
                im_original if background else None,
                show_cracks,
                mask_upper,
                mask_lower,
                mask_middle,
                frame_cracks,
                idx,
            )
        return [area_upper, area_lower, area_diffuse]

    def detect_all(
        self,
        plot: bool = False,
        background_plot: bool = False,
        show_cracks: bool = True,
        th_upper: Optional[float] = None,
        th_lower: Optional[float] = None,
        th_middle: Optional[float] = None,
        th_offset_upper: float = 0.0,
        th_offset_lower: float = 0.0,
        th_offset_middle: float = 0.0,
        th_alpha_upper: float = 1.0,
        th_alpha_lower: float = 1.0,
        th_alpha_middle: float = 1.0,
        use_last_frame_threshold: bool = False,
    ) -> List[List[float]]:
        """Run delamination detection over all frames.

        For each frame, this method computes areas (in pixels) and converts them into
        your existing real-world units using `area / scale_px_mm`. It returns per-frame
        rows with both absolute and relative areas.

        Args:
            plot: Whether to plot overlays for each frame.
            background_plot: If True, include specimen background in plots.
            show_cracks: If True, overlay cracks on plots.
            th_upper: Manual threshold for upper edge applied to all frames (optional).
            th_lower: Manual threshold for lower edge applied to all frames (optional).
            th_middle: Manual threshold for diffuse region applied to all frames (optional).
            th_offset_upper: Offset added to auto/last-frame threshold for upper edge.
            th_offset_lower: Offset added to auto/last-frame threshold for lower edge.
            th_offset_middle: Offset added to auto/last-frame threshold for diffuse region.
            th_alpha_upper: Multiplier applied to auto/last-frame threshold for upper edge.
            th_alpha_lower: Multiplier applied to auto/last-frame threshold for lower edge.
            th_alpha_middle: Multiplier applied to auto/last-frame threshold for diffuse region.
            use_last_frame_threshold: If True and manual thresholds are None, base auto-thresholds on last frame.

        Returns:
            List[List[float]]: For each frame:
                [idx, total_edge_mm2, upper_mm2, lower_mm2, edge_rel, diffuse_mm2, diffuse_rel]

        Notes:
            The conversion uses your existing convention `area / scale_px_mm`.
            Physically, area usually scales with the square of pixel-to-mm conversion,
            but this function preserves your current semantics for compatibility.

        Examples:
            Use manual thresholds for edges and an offset for the diffuse region:

            >>> results = detector.detect_all(
            ...     plot=True, background_plot=False,
            ...     th_upper=145, th_lower=145, th_offset_middle=-25.0
            ... )
            >>> len(results) > 0
            True

            Use last-frame Otsu as base threshold and tweak via alpha/offsets:

            >>> results = detector.detect_all(
            ...     use_last_frame_threshold=True,
            ...     th_alpha_upper=0.95, th_offset_upper=-5.0,
            ...     th_alpha_lower=1.05, th_offset_lower=+3.0,
            ...     th_alpha_middle=1.00, th_offset_middle=0.0,
            ... )
        """
        def compute_threshold(
            image: NDArray[np.uint8],
            manual_th: Optional[float],
            offset: float,
            alpha: float,
            base_th: Optional[float] = None,
            window_for_auto: Optional[Tuple[int, int]] = None,
        ) -> float:
            """Compute the threshold for a region.

            Args:
                image: Image used for auto-threshold setting.
                manual_th: Manual threshold; if provided, returned directly.
                offset: Additive offset applied to base/auto threshold.
                alpha: Multiplicative factor applied to base/auto threshold.
                base_th: Optional base threshold (e.g., from last frame).
                window_for_auto: Pre-filter window for auto thresholding.

            Returns:
                float: Threshold to use.
            """
            if manual_th is not None:
                return float(manual_th)
            th0 = base_th if base_th is not None else self.images_threshold(image, window_for_auto)
            return float(alpha) * float(th0) + float(offset)

        # Determine number of frames using the first available region
        region_lists = [
            getattr(self.specimen, "spec_middle", None),
            getattr(self.specimen, "spec_lower", None),
            getattr(self.specimen, "spec_upper", None),
            getattr(self.specimen, "spec_original", None),
        ]
        num_frames = None
        for region in region_lists:
            if region is not None and hasattr(region, "image_paths"):
                num_frames = len(region.image_paths)
                break
        if num_frames is None:
            raise ValueError("No valid region with image_paths found in specimen.")

        results: List[List[float]] = []

        # Base thresholds (optionally from last frame)
        base_th_upper = None
        base_th_lower = None
        base_th_middle = None

        if use_last_frame_threshold:
            if th_upper is None:
                im_upper_last = imread(self.specimen.spec_upper.image_paths[-1])
                base_th_upper = self.images_threshold(im_upper_last, self.window_size_edge)
            if th_lower is None:
                im_lower_last = imread(self.specimen.spec_lower.image_paths[-1])
                base_th_lower = self.images_threshold(im_lower_last, self.window_size_edge)
            if th_middle is None:
                im_middle_last = imread(self.specimen.spec_middle.image_paths[-1])
                base_th_middle = self.images_threshold(im_middle_last, self.window_size_diffuse)

        # Initialize previous-fill buffers if needed
        self._ensure_prev_buffers()

        im0 = imread(self.specimen.spec_original.image_paths[0])
        H, W = im0.shape[:2]
        specimen_area = (H * W) / float(self.specimen.scale_px_mm)

        for idx in range(num_frames):
            im_u = imread(self.specimen.spec_upper.image_paths[idx])
            im_l = imread(self.specimen.spec_lower.image_paths[idx])
            im_m = imread(self.specimen.spec_middle.image_paths[idx])

            th_u = compute_threshold(im_u, th_upper, th_offset_upper, th_alpha_upper, base_th_upper, self.window_size_edge)
            th_l = compute_threshold(im_l, th_lower, th_offset_lower, th_alpha_lower, base_th_lower, self.window_size_edge)
            th_m = compute_threshold(im_m, th_middle, th_offset_middle, th_alpha_middle, base_th_middle, self.window_size_diffuse)

            area_u_px, area_l_px, area_m_px = self.detect(
                idx=idx,
                plot=plot,
                background=background_plot,
                th_upper=th_u,
                th_lower=th_l,
                th_middle=th_m,
                show_cracks=show_cracks,
            )

            # Convertsion to "real" units from px**2 to mm**2
            real_u = area_u_px / float(self.specimen.scale_px_mm)**2
            real_l = area_l_px / float(self.specimen.scale_px_mm)**2
            real_m = area_m_px / float(self.specimen.scale_px_mm)**2

            total_edge = real_u + real_l
            rel_edge = total_edge / specimen_area if specimen_area > 0 else 0.0
            rel_diff = real_m / specimen_area if specimen_area > 0 else 0.0

            results.append([idx, total_edge, real_u, real_l, rel_edge, real_m, rel_diff])

        return results
