"""
Delamination detection utilities for DelaDect.

Segments edge and diffuse delamination in specimen image stacks produced by the crack detection pipeline, and provides helpers to threshold, post-process, and plot the resulting masks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    from numpy.typing import NDArray
except (ImportError, AttributeError):
    NDArray = np.ndarray  # type: ignore[attr-defined]
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.io import imread

# Imported for type context; no runtime dependency on this symbol’s contents.
from deladect.detection import Specimen


# Utilities


def _as_bool_mask(mask: NDArray) -> NDArray[np.bool_]:
    """Convert an array to a boolean mask (True where values > 0).

    Args:
        mask: Input array of any numeric dtype.

    Returns:
        Boolean mask where True indicates positive values.
    """
    return np.asarray(mask) > 0


def _ensure_uint8(image: NDArray) -> NDArray[np.uint8]:
    """Return a uint8 view of an image, rescaling floats when needed.

    Floats with max <= 1 are treated as normalised data and scaled to 0-255.
    Other dtypes are clipped into the byte range before casting.
    """
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        max_val = float(np.nanmax(arr)) if arr.size else 0.0
        scaled = arr * 255.0 if max_val <= 1.0 + 1e-6 else arr
        return np.clip(scaled, 0.0, 255.0).astype(np.uint8)
    return np.clip(arr, 0, 255).astype(np.uint8)

def edge_rows(mask: NDArray[np.uint8], side: str) -> Optional[NDArray[np.float64]]:
    """Return the first or last positive row per column for a binary mask.
    
    `side` selects the traversal direction; columns without detections yield NaN.
    Returns None when the mask has no hits.
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
    """Plot the edge profile reconstructed by `edge_rows` onto `ax`."""
    if rows_local is None or start_row is None:
        return
    x = np.arange(cols_w)
    y = start_row + rows_local  # rows_local may include NaNs
    ax.plot(x, y, **kwargs)


def to_binary_mask(arr: Optional[NDArray]) -> Optional[NDArray[np.uint8]]:
    """Return a binary uint8 mask (0/1); propagate None inputs."""
    if arr is None:
        return None
    return (_as_bool_mask(arr)).astype(np.uint8)


def has_hits(arr: Optional[NDArray]) -> bool:
    """Return True when the array exists and contains a positive value."""
    return arr is not None and arr.size > 0 and np.any(arr)


# Detector

class DelaminationDetector:
    """Compute delamination masks and measurements for a specimen.
    
    Handles upper/lower edge slices and diffuse regions around cracks.
    Thresholding, smoothing, ROI padding, and optional frame accumulation
    are configured via the constructor.
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
          5. Vertical minimum accumulate (column-wise) for edge delamination

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

    def _compute_diffuse_bounds(
        self,
        crack: NDArray,
        image_shape: Tuple[int, int],
    ) -> Tuple[int, int, int, int]:
        """Return clamped ROI bounds (y0, y1, x0, x1) around a crack."""
        h, w = image_shape
        (y1, x1), (y2, x2) = crack
        dx = float(self.diffuse_dx)
        dy = float(self.diffuse_dy)
        x_lo = int(max(0, min(x1, x2) - dx))
        x_hi = int(min(w, max(x1, x2) + dx))
        y_lo = int(max(0, min(y1, y2) - dy))
        y_hi = int(min(h, max(y1, y2) + dy))
        return y_lo, y_hi, x_lo, x_hi

    def _diffuse_roi_bounds(
        self,
        crack: NDArray,
        image_shape: Tuple[int, int],
    ) -> Optional[Tuple[int, int, int, int]]:
        """Bounds helper that discards empty ROIs."""
        y_lo, y_hi, x_lo, x_hi = self._compute_diffuse_bounds(crack, image_shape)
        if x_hi <= x_lo or y_hi <= y_lo:
            return None
        return y_lo, y_hi, x_lo, x_hi

    def _process_diffuse_roi(
        self,
        image: NDArray[np.uint8],
        bounds: Tuple[int, int, int, int],
        th: Optional[float],
        min_pixel_value: int,
    ) -> NDArray[np.uint8]:
        """Run the diffuse filtering pipeline on a ROI and return its mask."""
        y_lo, y_hi, x_lo, x_hi = bounds
        roi = image[y_lo:y_hi, x_lo:x_hi]
        if roi.size == 0:
            return np.zeros((0, 0), dtype=np.uint8)
        if th is None:
            th_value = self.images_threshold(roi, self.window_size_diffuse)
        else:
            th_value = float(th)
        filtered_max = ndimage.maximum_filter(roi, size=self.window_size_diffuse)
        filtered_min = ndimage.minimum_filter(filtered_max, size=self.window_size_diffuse)
        binary255 = (filtered_min < th_value).astype(np.uint8) * 255
        smoothed = ndimage.gaussian_filter(binary255, self.gaussian_filters)
        return (smoothed > int(min_pixel_value)).astype(np.uint8)

    def _prepare_edge_slice(self, image: NDArray[np.uint8], im_type: str) -> NDArray[np.uint8]:
        """Return the edge slice in processing orientation (upper stays, lower flips)."""
        return np.flipud(image) if im_type == "Lower" else image

    def _resolve_edge_threshold(self, image: NDArray[np.uint8], manual_th: Optional[float]) -> float:
        """Return the threshold for an edge slice, defaulting to auto-Otsu."""
        return self.images_threshold(image) if manual_th is None else float(manual_th)

    def _edge_mask(self, image: NDArray[np.uint8], threshold: float) -> NDArray[np.uint8]:
        """Run the edge filtering pipeline for a prepared slice."""
        return self.apply_filters(image, threshold, self.window_size_edge)

    def _diffuse_mask_for_crack(
        self,
        image: NDArray[np.uint8],
        crack: NDArray,
        manual_th: Optional[float],
        min_pixel_value: int,
    ) -> Tuple[NDArray[np.uint8], Tuple[int, int, int, int]]:
        """Return mask and bounds for a single crack ROI."""
        h, w = image.shape[:2]
        y_lo, y_hi, x_lo, x_hi = self._compute_diffuse_bounds(crack, (h, w))
        bounds = (y_lo, y_hi, x_lo, x_hi)
        if x_hi <= x_lo or y_hi <= y_lo:
            return np.zeros((0, 0), dtype=np.uint8), bounds
        roi_mask = self._process_diffuse_roi(image, bounds, manual_th, min_pixel_value)
        return roi_mask, bounds


    def images_threshold(
        self,
        image: NDArray[np.uint8],
        window_size: Optional[Tuple[int, int]] = None,
    ) -> float:
        """Compute an Otsu threshold after pre-filtering.

        A max filter first dilates cracks and delaminated regions,
        and a min filter then erodes isolated "hot" pixels. The pair acts like a
        closing operation so the histogram presented to Otsu is smoother and more
        bimodal, yielding a more stable threshold on noisy image samples.
        """
        if window_size is None:
            window_size = self.window_size_edge
        wy, wx = max(1, int(window_size[0])), max(1, int(window_size[1]))
        filtered_max = ndimage.maximum_filter(image, size=(wy, wx))
        filtered_min = ndimage.minimum_filter(filtered_max, size=(wy, wx))
        return float(threshold_otsu(filtered_min))

    def edge_delamination_image(
        self,
        image: NDArray[np.uint8],
        im_type: str = "Upper",
        th: Optional[float] = None,
    ) -> NDArray[np.uint8]:
        """Run the edge filtering pipeline on an upper or lower slice.

        Uses auto-Otsu when `th` is not provided and applies the shared edge
        helpers for consistent processing.
        """
        slice_img = self._prepare_edge_slice(image, im_type)
        threshold = self._resolve_edge_threshold(slice_img, th)
        return self._edge_mask(slice_img, threshold)

    # Diffuse delamination
    def diffuse_delamination_image(
        self,
        image: NDArray[np.uint8],
        cracks: Optional[NDArray],
        th: Optional[float] = None,
        min_pixel_value: int = 160,
    ) -> NDArray[np.uint8]:
        """Generate the diffuse delamination mask for a frame.

        Processes each crack ROI individually and unions the resulting masks.
        `min_pixel_value` sets the Gaussian-response cutoff (160 keeps legacy behaviour).
        """
        h, w = image.shape[:2]
        output_image = np.zeros((h, w), dtype=np.uint8)

        if not (cracks is not None and getattr(cracks, "size", 0) > 0):
            return output_image

        manual_th = None if th is None else float(th)

        for crack in cracks:
            roi_mask, bounds = self._diffuse_mask_for_crack(
                image, crack, manual_th, min_pixel_value
            )
            y_lo, y_hi, x_lo, x_hi = bounds
            if roi_mask.size == 0:
                continue
            output_image[y_lo:y_hi, x_lo:x_hi] = np.maximum(
                output_image[y_lo:y_hi, x_lo:x_hi],
                roi_mask,
            )

        return output_image


    def diffuse_delamination_each_crack(
        self,
        image: NDArray[np.uint8],
        crack: NDArray,
        th: float,
        min_pixel_value: int = 160,
    ) -> Tuple[NDArray[np.uint8], int, int, int, int]:
        """Return the diffuse mask for a single crack ROI along with its bounds."""
        roi_mask, (y_lo, y_hi, x_lo, x_hi) = self._diffuse_mask_for_crack(
            image, crack, float(th), min_pixel_value
        )
        return roi_mask, y_lo, y_hi, x_lo, x_hi


    # Areas

    @staticmethod
    def calculate_area(filtered_image: NDArray[np.uint8]) -> float:
        """Count non-zero pixels in a binary mask."""
        return float(np.count_nonzero(filtered_image))

    # Crack retrieval

    def get_frame_cracks(self, frame_idx: int) -> NDArray:
        """Collect cracks for `frame_idx` across all orientations.
        
        Prefers `cracks_original` when available; returns an empty array if no crack data is stored.
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

    # Plotting

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
        """Save overlay visualisations for the current frame.
        
        Optionally draws the grayscale background and detected cracks before
        exporting the masks to `output_dir`.
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
            im0 = self.specimen.image_stack_upper[0]
            self._prev_upper = np.zeros_like(im0, dtype=np.uint8)
        if self._prev_middle is None:
            im0 = self.specimen.image_stack_middle[0]
            self._prev_middle = np.zeros_like(im0, dtype=np.uint8)
        if self._prev_lower is None:
            im0 = self.specimen.image_stack_lower[0]
            self._prev_lower = np.zeros_like(im0, dtype=np.uint8)

    def detect(
        self,
        idx: int = 0,
        plot: bool = False,
        background: bool = False,
        th_upper: Optional[float] = None,
        th_lower: Optional[float] = None,
        th_middle: Optional[float] = None,
        min_pixel_value_middle: int = 160,
        show_cracks: bool = True,
        dir_output: str = "output_images",
    ) -> List[float]:
        """Run delamination detection for a single frame.

        Args:
            idx: Frame index to analyze.
            plot: Whether to plot and save overlays for this frame.
            background: If True, include the original specimen background in the plot.
            th_upper: Manual threshold for upper edge (optional).
            th_lower: Manual threshold for lower edge (optional).
            th_middle: Manual threshold for diffuse region (optional).
            min_pixel_value_middle: Post-smoothing cutoff for diffuse masks (optional).
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
        im_original = _ensure_uint8(self.specimen.image_stack_cut[idx])
        im_upper = _ensure_uint8(self.specimen.image_stack_upper[idx])
        im_lower = _ensure_uint8(self.specimen.image_stack_lower[idx])
        im_middle = _ensure_uint8(self.specimen.image_stack_middle[idx])

        # cracks
        frame_cracks = self.get_frame_cracks(idx)

        # build masks
        mask_upper = self.edge_delamination_image(im_upper, im_type="Upper", th=th_upper)
        mask_lower = self.edge_delamination_image(im_lower, im_type="Lower", th=th_lower)
        mask_middle = self.diffuse_delamination_image(
            im_middle, frame_cracks, th=th_middle, min_pixel_value=min_pixel_value_middle
        )

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
                output_dir=dir_output
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
        min_pixel_value_middle: int = 160,
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
            min_pixel_value_middle: Post-smoothing cutoff for diffuse masks (optional).
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
                min_pixel_value_middle=min_pixel_value_middle,
                show_cracks=show_cracks,
            )

            # Converts to "real" units from px**2 to mm**2
            real_u = area_u_px / float(self.specimen.scale_px_mm)**2
            real_l = area_l_px / float(self.specimen.scale_px_mm)**2
            real_m = area_m_px / float(self.specimen.scale_px_mm)**2

            total_edge = real_u + real_l
            rel_edge = total_edge / specimen_area if specimen_area > 0 else 0.0
            rel_diff = real_m / specimen_area if specimen_area > 0 else 0.0

            results.append([idx, total_edge, real_u, real_l, rel_edge, real_m, rel_diff])

        return results



