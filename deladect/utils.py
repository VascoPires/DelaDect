"""
Utilities for crack detection and specimen image/data management in DelaDect.
Provides helper functions and classes for crack geometry, scaling, and data organization.
"""

from pathlib import Path
import pickle
from functools import lru_cache
import numpy as np
from typing import List, Optional, Union

def crack_mid_point(crack: List[List[float]]) -> List[float]:
    """Return the midpoint of a crack segment.

    Args:
        crack: Sequence with two endpoints ``[[x0, y0], [x1, y1]]``.

    Returns:
        list[float]: Midpoint coordinates or ``[None, None]`` when the input is malformed.

    Example:
        >>> crack_mid_point([[0, 0], [4, 2]])  # doctest: +NORMALIZE_WHITESPACE
        [2.0, 1.0]
    """

    crack_array = np.array(crack)
    
    # Check if crack is empty or doesn't have exactly 2 points with 2 coordinates each
    if crack_array.size == 0 or crack_array.shape != (2, 2):
        return [None, None]
    mid_x = (crack_array[0][0] + crack_array[1][0]) / 2
    mid_y = (crack_array[0][1] + crack_array[1][1]) / 2
    return [mid_x, mid_y]

def crack_length(crack: List[List[float]]) -> float:
    """Compute the Euclidean length of a crack segment.

    Args:
        crack: Sequence with two ``[x, y]`` coordinates describing the endpoints.

    Returns:
        float: Segment length in pixels; ``0.0`` when the input is invalid.

    Example:
        >>> round(crack_length([[0, 0], [3, 4]]), 1)
        5.0
    """
    if crack is None or len(crack) != 2:
        return 0.0
    (x_0, y_0), (x_1, y_1) = crack
    length = ((x_1 - x_0)**2 + (y_1 - y_0)**2)**0.5
    return length

def crack_px_mm(crack_list: List[List[List[float]]], scale: float) -> List[List[List[float]]]:
    """Convert crack coordinates from pixels to millimetres.

    Args:
        crack_list: Nested list of crack endpoints in pixels.
        scale: Scaling factor expressed as pixels per millimetre.

    Returns:
        list: Crack coordinates converted to millimetres; the original list is returned when
        ``scale`` evaluates to ``0``.

    Example:
        >>> crack_px_mm([[[0, 0], [10, 0]]], scale=5)
        [[[0.0, 0.0], [2.0, 0.0]]]
    """
    if not crack_list or scale == 0:
        return crack_list
    scaled_crack = [[[x / scale, y / scale] for x, y in segment] for segment in crack_list]
    return scaled_crack

class SpecimenImages:
    """Discover and sort image files for a specimen region.

    Parameters:
        path: Directory containing the images of interest.
        suffix: Optional file suffix to filter images (for example ``'.png'``).

    Attributes:
        image_paths: Sorted list of resolved image paths.
    """
    def __init__(self, path: Union[str, Path], suffix: Optional[str] = None):
        self.image_paths = self._load_image_paths(path, suffix)

    @staticmethod
    def _load_image_paths(path: Union[str, Path], suffix: Optional[str] = None) -> List[str]:
        """Load image paths from ``path`` and sort them numerically when possible.

        Args:
            path: Directory containing the images.
            suffix: Optional suffix used to filter the discovered files.

        Returns:
            list[str]: Sorted list of image paths ready for consumption by ``crackdect``.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image directory not found: {path}")
        # Use suffix if provided, otherwise accept all images
        if suffix:
            files = sorted(path.glob(f"*{suffix}"))
        else:
            files = sorted(path.glob("*"))
        # Try to sort numerically if possible
        def numeric_key(p):
            stem = p.stem
            try:
                return int(''.join(filter(str.isdigit, stem)))
            except ValueError:
                return stem
        return [str(p) for p in sorted(files, key=numeric_key)]

class DataProcessor:
    """Utility helpers for organising output folders and serialised artefacts.

    The processor centralises the logic used across the project to create folder
    structures and persist crack metadata.
    """
    def __init__(self, base_folder: Optional[Union[str, Path]] = None):
        self.data_folder_path = self._get_data_folder_path(base_folder)

    @staticmethod
    def _get_data_folder_path(base_folder: Optional[Union[str, Path]] = None) -> str:
        """Resolve the base data directory.

        Args:
            base_folder: Optional base path overriding the default ``./data`` directory.

        Returns:
            str: Absolute path to the folder that should contain exported artefacts.
        """
        if base_folder:
            folder_path = Path(base_folder).resolve()
        else:
            folder_path = Path.cwd() / "data"
        return str(folder_path)

    @staticmethod
    def list_folders(path: Union[str, Path]) -> List[str]:
        """Return the immediate subdirectories inside ``path``.

        Args:
            path: Directory to inspect.

        Returns:
            list[str]: Absolute paths for each immediate subdirectory.
        """
        path = Path(path)
        return [str(p) for p in path.iterdir() if p.is_dir()]

    @staticmethod
    def generate_folder(parent_folder: Optional[Union[str, Path]] = None, sub_folder: Optional[str] = None) -> str:
        """Ensure a directory structure exists and return its path.

        Args:
            parent_folder: Base directory that should contain the artefacts.
            sub_folder: Optional subdirectory name to create inside ``parent_folder``.

        Returns:
            str: Absolute path to the created directory.

        Example:
            >>> DataProcessor.generate_folder('output', 'cracks')  # doctest: +SKIP
            '.../output/cracks'
        """
        if parent_folder:
            parent_folder = Path(parent_folder)
            parent_folder.mkdir(parents=True, exist_ok=True)
        if sub_folder:
            folder_path = parent_folder / sub_folder if parent_folder else Path(sub_folder)
            folder_path.mkdir(parents=True, exist_ok=True)
            return str(folder_path)
        return str(parent_folder) if parent_folder else ""

    @staticmethod
    def save_cracks_to_file(cracks, folder_name: Union[str, Path], file_name: str):
        """Serialise cracks to disk using :mod:`pickle`.

        Args:
            cracks: Crack data to persist.
            folder_name: Directory that should contain the serialised file.
            file_name: File name to use for the pickle.

        Returns:
            None: The data is written to ``folder_name / file_name`` and the location is printed.
        """
        folder_path = Path(folder_name)
        folder_path.mkdir(parents=True, exist_ok=True)
        file_path = folder_path / file_name
        with open(file_path, 'wb') as f:
            pickle.dump(cracks, f)
        print(f"Cracks saved to file: {file_path}")