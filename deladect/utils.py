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
    """
    Calculate the midpoint of a singular crack.

    Args:
        crack (list): List containing two crack coordinates [[x0, y0], [x1, y1]].

    Returns:
        list: Coordinates of the midpoint of the crack, or [None, None] if crack is empty.
    """

    crack_array = np.array(crack)
    
    # Check if crack is empty or doesn't have exactly 2 points with 2 coordinates each
    if crack_array.size == 0 or crack_array.shape != (2, 2):
        return [None, None]
    mid_x = (crack_array[0][0] + crack_array[1][0]) / 2
    mid_y = (crack_array[0][1] + crack_array[1][1]) / 2
    return [mid_x, mid_y]

def crack_length(crack: List[List[float]]) -> float:
    """
    Calculate the length of a singular crack.

    Args:
        crack (list or np.ndarray): List or array containing two crack coordinates [[x0, y0], [x1, y1]].

    Returns:
        float: Length of the crack, or 0.0 if crack is empty.
    """
    if crack is None or len(crack) != 2:
        return 0.0
    (x_0, y_0), (x_1, y_1) = crack
    length = ((x_1 - x_0)**2 + (y_1 - y_0)**2)**0.5
    return length

def crack_px_mm(crack_list: List[List[List[float]]], scale: float) -> List[List[List[float]]]:
    """
    Scale a list of cracks from pixels to millimeters.

    Args:
        crack_list (list): List containing crack coordinates in pixels.
        scale (float): Scaling factor from pixels to millimeters.

    Returns:
        list: Scaled crack coordinates in millimeters.
    """
    if not crack_list or scale == 0:
        return crack_list
    scaled_crack = [[[x / scale, y / scale] for x, y in segment] for segment in crack_list]
    return scaled_crack

class SpecimenImages:
    """
    Handles loading and accessing images from a specified directory.

    Args:
        path (str or Path): Directory containing the images.
        suffix (str): Optional file suffix to filter images (e.g., '.png').
    """
    def __init__(self, path: Union[str, Path], suffix: Optional[str] = None):
        self.image_paths = self._load_image_paths(path, suffix)

    @staticmethod
    def _load_image_paths(path: Union[str, Path], suffix: Optional[str] = None) -> List[str]:
        """
        Load and sort image paths from the specified directory.

        Args:
            path: Directory containing the images.
            suffix: Optional file suffix to filter images.

        Returns:
            A sorted list of image paths.
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
    """
    Handles data organization and folder management.
    """
    def __init__(self, base_folder: Optional[Union[str, Path]] = None):
        self.data_folder_path = self._get_data_folder_path(base_folder)

    @staticmethod
    def _get_data_folder_path(base_folder: Optional[Union[str, Path]] = None) -> str:
        """
        Get the path to the data folder.

        Args:
            base_folder: Optional base folder path.

        Returns:
            The absolute path to the data folder.
        """
        if base_folder:
            folder_path = Path(base_folder).resolve()
        else:
            folder_path = Path.cwd() / "data"
        return str(folder_path)

    @staticmethod
    def list_folders(path: Union[str, Path]) -> List[str]:
        """
        List all folders in the specified path.

        Args:
            path: Directory to search for folders.

        Returns:
            A list of folder paths.
        """
        path = Path(path)
        return [str(p) for p in path.iterdir() if p.is_dir()]

    @staticmethod
    def generate_folder(parent_folder: Optional[Union[str, Path]] = None, sub_folder: Optional[str] = None) -> str:
        """
        Create a folder structure.

        Args:
            parent_folder: Parent folder path.
            sub_folder: Subfolder name.

        Returns:
            The path to the created subfolder.
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
        """
        Save the list of cracks to a file using pickle.

        Args:
            cracks: The list of cracks to save.
            folder_name: The directory where the file will be saved.
            file_name: The name of the file.
        """
        folder_path = Path(folder_name)
        folder_path.mkdir(parents=True, exist_ok=True)
        file_path = folder_path / file_name
        with open(file_path, 'wb') as f:
            pickle.dump(cracks, f)
        print(f"Cracks saved to file: {file_path}")