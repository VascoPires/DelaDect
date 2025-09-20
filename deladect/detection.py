"""
Detection module for DelaDect: Provides the Specimen class and methods for crack detection, evaluation, and post-processing.
Allows to compute crack density, crack spacing and plotting of cracks.
Integrates with the crackdect library for image-based crack analysis in composite materials.
"""

from pathlib import Path
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import csv
import time

from typing import List, Tuple, Dict, Optional, Any
from crackdect import detect_cracks_bender, ImageStack, ImageStackSQL, image_paths, sort_paths
from skimage.io import imread
from deladect.utils import DataProcessor, crack_length, crack_mid_point


class Specimen:
    """
    A class to represent a specimen and perform various operations related to it.
    """
    def __init__(self, name: str, dimensions: dict, scale_px_mm: float,
                 path_cut: str, path_upper_border: str, path_lower_border: str, path_middle: str,
                 sorting_key: str, image_types: list, avg_crack_width: float = 10.0,
                 strain_csv: Optional[str] = None, 
                 stack_backend: str = "auto", 
                 stack_limit_mb: float = 512.0, 
                 sql_stack_kwargs: Optional[Dict[str, Any]] = None):
        self.name = name
        self.dimensions = dimensions  # Dictionary containing width, thickness, etc.
        self.scale_px_mm = scale_px_mm
        self.path_cut = path_cut
        self.path_upper_border = path_upper_border
        self.path_lower_border = path_lower_border
        self.path_middle = path_middle
        self.sorting_key = sorting_key
        self.image_types = image_types
        backend = (stack_backend or "auto").lower()
        if backend not in {"auto", "memory", "sql"}:
            raise ValueError('stack_backend must be "auto", "memory", or "sql"')
        self._stack_backend = backend
        self._stack_limit_bytes = max(float(stack_limit_mb), 0.0) * 1024 * 1024
        self._sql_stack_kwargs = dict(sql_stack_kwargs or {})

        # Create the ImageStack for the Cut or Original image
        self.path_cut_list = list(image_paths(self.path_cut, image_types=self.image_types))
        self.path_cut_list, self.cycles_cut = sort_paths(self.path_cut_list, sorting_key=self.sorting_key)
        self.path_cut_list = list(self.path_cut_list)
        self.image_stack_cut = self._build_stack(self.path_cut_list, dtype=np.float32, as_gray=True)

        # Create the ImageStack for the middle region
        self.path_middle_list = list(image_paths(self.path_middle, image_types=self.image_types))
        self.path_middle_list, self.cycles_middle = sort_paths(self.path_middle_list, sorting_key=self.sorting_key)
        self.path_middle_list = list(self.path_middle_list)
        self.image_stack_middle = self._build_stack(self.path_middle_list, dtype=np.float32, as_gray=True)


        self.avg_crack_width_px = avg_crack_width
        
        if strain_csv is not None:
            df = pd.read_csv(strain_csv)
            self.experimental_data = df[['strain_y']].reset_index(drop=True)
        else:
            self.experimental_data = None

    @staticmethod
    def _estimate_stack_bytes(paths: List[str], *, dtype=np.float32, as_gray: Optional[bool] = True) -> int:
        paths = list(paths)
        if not paths:
            return 0
        sample = imread(paths[0], as_gray=as_gray) if as_gray is not None else imread(paths[0])
        arr = np.asarray(sample, dtype=dtype) if dtype is not None else np.asarray(sample)
        return int(arr.nbytes) * len(paths)

    def _build_stack(self, paths: List[str], *, dtype=np.float32, as_gray: Optional[bool] = True):
        paths = list(paths)
        if not paths:
            raise ValueError('Cannot build an image stack without any image paths.')
        backend = self._stack_backend
        selected = backend
        if backend == 'auto':
            limit = self._stack_limit_bytes
            est_bytes = self._estimate_stack_bytes(paths, dtype=dtype, as_gray=as_gray)
            if limit > 0 and est_bytes > limit:
                selected = 'sql'
            else:
                selected = 'memory'
        if selected == 'sql':
            kwargs = dict(self._sql_stack_kwargs)
            if dtype is not None and 'dtype' not in kwargs:
                kwargs['dtype'] = dtype
            if as_gray is not None and 'as_gray' not in kwargs:
                kwargs['as_gray'] = as_gray
            return ImageStackSQL.from_paths(paths, **kwargs)
        if selected != 'memory':
            raise ValueError("Unsupported stack backend '{}'".format(selected))
        kwargs = {}
        if dtype is not None:
            kwargs['dtype'] = dtype
        if as_gray is not None:
            kwargs['as_gray'] = as_gray
        return ImageStack.from_paths(paths, **kwargs)
        

    def crack_eval(self, theta_fd: int, crack_w: Optional[float] = None,
                   min_crack_size: Optional[float] = None,
                   export_images: bool = False,
                   background: bool = False, 
                   comparison: bool = False,
                   save_cracks: bool = False,
                   image_stack_orig = False,
                   color_cracks = 'red',
                   output_dir: Optional[str] = None) -> Tuple[List[np.ndarray], List[float], List[float]]:
        """
        Evaluate cracks in an image stack and optionally export images.

        Args:
            theta_fd: Crack orientation angle.
            crack_w: Crack width (in px). Defaults to self.avg_crack_width_px if not provided.
            min_crack_size: Minimum crack size (in px).
            export_images: Whether to export identified crack images.
            comparison: Whether to include comparison in the plots.

        Returns:
            A tuple containing cracks, rho, and theta values.
        """
        # Use default crack width if not provided
        if crack_w is None:
            crack_w = self.avg_crack_width_px
        if min_crack_size is None:
            min_crack_size = self.dimensions['width'] * self.scale_px_mm * 0.10
        if crack_w is None or min_crack_size is None:
            raise ValueError("crack_w and min_crack_size must not be None.")


        # Detect cracks
        if image_stack_orig:
            rho, cracks, th = detect_cracks_bender(
                self.image_stack_cut,
                theta=int(theta_fd),
                crack_width=int(crack_w),
                min_size=int(min_crack_size)
            )
        else:
            rho, cracks, th = detect_cracks_bender(
                self.image_stack_middle,
                theta=int(theta_fd),
                crack_width=int(crack_w),
                min_size=int(min_crack_size)
            )
            

        if export_images:
             
            # Generate folders
            output_dir = output_dir or "Crack Detection"
            self.identified_cracks_subfolder = DataProcessor.generate_folder(output_dir, "Identified Cracks")
            sample_subfolder = DataProcessor.generate_folder(self.identified_cracks_subfolder, f"{self.name}__+{theta_fd}")
        
            for idx, crack in enumerate(cracks):
                fig, ax = self.plot_cracks(self.image_stack_middle[idx], crack,
                                           background_flag=background,
                                           color=color_cracks,
                                           comparison=comparison)
                ax.set_xlabel('x [Px]')
                ax.set_ylabel('y [Px]')
                filename = str(Path(sample_subfolder) / f"Cracks_{idx}.png")
                fig.savefig(filename)
                plt.close(fig)

        if save_cracks:
            DataProcessor.save_cracks_to_file(cracks, sample_subfolder, f"{self.name}_cracks_data_{int(theta_fd)}.pkl")

        return cracks, [float(r) for r in rho], [float(t) for t in th]

    @staticmethod
    def plot_cracks(image, cracks, linewidth=1, color='red',
                    background_flag = False, 
                    comparison=False, ax = None, **kwargs):
        """
        Plots cracks in the foreground with an option of displaying the background.
            Note: This function is directly adapted from CrackDect library https://doi.org/10.1016/j.softx.2021.100832.
        
        Parameters
        ----------
        image: np.ndarray
            Background image
        cracks: np.ndarray
            Array with the coordinates of the crack with the following structure:
            ([[x0, y0],[x1,y1], [...]]) where x0 and y0 are the starting coordinates and x1, y1
            the end of one crack. Each crack is represented by a 2x2 array stacked into a bigger array (x,2,2).
        kwargs:
            Forwarded to `plt.figure() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html>`_

        Returns
        -------
        fig: Figure
        ax: Axes
        """
        if ax is None:
            fig = plt.figure(**kwargs)
            ax = fig.add_subplot(111)
        else:
            fig = ax.figure  # Get the figure from the axis

        if comparison:
            image = np.hstack((image, image))
        if background_flag:
            ax.imshow(image, cmap='gray')

        if cracks is not None and isinstance(cracks, (list, np.ndarray)) and len(cracks) > 0:
            for (y0, x0), (y1, x1) in cracks:
                ax.plot((x0, x1), (y0, y1), color=color, linewidth=linewidth,linestyle='-')

        ax.set_ylim(image.shape[0], 0)
        ax.set_xlim(0, image.shape[1])
        ax.set_aspect('equal')
        ax.tick_params(axis='both', which='both', length=0)
        return fig, ax


    def export_rho(
        self,
        *rho_lists: List[float],
        folder_name: str = "Crack Detection",
        file_name: str = "rho_data.csv",
        rho_names: Optional[List[str]] = None
    ) -> None:
        """
        Export multiple rho lists to a CSV file, each as a separate column.
        Optionally merges experimental 'strain_y' data if available.

        Args:
            *rho_lists: Variable number of lists of rho values.
            folder_name: Directory to save the file.
            file_name: Name of the CSV file.
            rho_names: Optional list of column names for each rho list.
        """
        if not rho_lists or any(len(rho) == 0 for rho in rho_lists):
            print("No rho data to export.")
            return

        num_entries = len(rho_lists[0])
        if not all(len(rho) == num_entries for rho in rho_lists):
            raise ValueError("All rho lists must have the same length.")

        # Default column names if not provided
        if rho_names is None:
            rho_names = [f"rho_{i+1}" for i in range(len(rho_lists))]

        # Build DataFrame
        data = {'ID': range(num_entries)}
        for name, rho in zip(rho_names, rho_lists):
            data[name] = rho
        df = pd.DataFrame(data)

        # Merge strain_y if experimental data has been uploaded
        if getattr(self, "experimental_data", None) is not None:
            if "strain_y" in self.experimental_data.columns:
                strain_y = self.experimental_data['strain_y'].reset_index(drop=True)
                df = pd.concat([df.iloc[:, :1], strain_y, df.iloc[:, 1:]], axis=1)

        # Save to CSV
        Path(folder_name).mkdir(parents=True, exist_ok=True)
        file_path = str(Path(folder_name) / file_name)
        df.to_csv(file_path, index=False)
        print(f"Multiple rho data exported to: {file_path}")
    
    def export_crack_spacing(
        self,
        processed_data: list,
        folder_name: str = "Crack Detection",
        file_name: str = "crack_spacing_data.csv"
    ) -> None:
        """
        Export crack spacing data to a CSV file. Handles both full and scaled dictionary formats.

        Args:
            processed_data: List of dictionaries with crack spacing data.
            folder_name: Directory to save the file.
            file_name: Name of the CSV file.
            experimental_data: Optional DataFrame to merge with crack spacing data.
        """
        if not processed_data:
            print("No data to export.")
            return

        # Determine which keys are present
        if all(('Crack_spacing' in d or 'Cracks_filtered' in d) for d in processed_data):
            # Full format
            columns = ['Picture', 'Avg_spacing', 'Std_spacing', 'Crack_spacing']
            rows = [
                {
                    'Picture': d.get('Picture', None),
                    'Avg_spacing': d.get('Avg_spacing', None),
                    'Std_spacing': d.get('Std_spacing', None),
                    'Crack_spacing': d.get('Crack_spacing', None)
                }
                for d in processed_data
            ]
        else:
            # Scaled/short format
            columns = ['Picture', 'Avg_spacing', 'Std_spacing']
            rows = [
                {
                    'Picture': d.get('Picture', None),
                    'Avg_spacing': d.get('Avg_spacing', None),
                    'Std_spacing': d.get('Std_spacing', None)
                }
                for d in processed_data
            ]

        df = pd.DataFrame(rows, columns=columns)  # type: ignore

        # If strain is available, merge it into the DataFrame
        if self.experimental_data is not None:

            strain_y = self.experimental_data['strain_y'].reset_index(drop=True)
            # Insert 'strain_y' as the second column
            df = pd.concat([df.iloc[:, :1], strain_y, df.iloc[:, 1:]], axis=1)

        Path(folder_name).mkdir(parents=True, exist_ok=True)
        file_path = str(Path(folder_name) / file_name)
        df.to_csv(file_path, index=False)
        print(f"Crack spacing data exported to: {file_path}")


    def order_cracks(
        self, 
        crack_list: np.ndarray, 
        delimiter: bool = True, 
        image_height: int = 1, 
        image_width: int = 1
    ) -> np.ndarray:
        """
        Order cracks based on the minimum y-coordinate of their endpoints.

        Args:
            crack_list: NumPy array of shape (N, 2, 2) representing cracks, where N is the number of cracks,
                        and each crack is represented by two endpoints.
            delimiter: Whether to include delimiter cracks at the image borders.
            image_height: Height of the image (in pixels).
            image_width: Width of the image (in pixels).

        Returns:
            Ordered NumPy array of cracks, optionally with delimiter cracks at the borders.
        """
        if len(crack_list) == 0:
            return crack_list

        # Calculate the minimum y-coordinate for each crack
        min_y_coords = np.minimum(crack_list[:, 0, 1], crack_list[:, 1, 1])
        ordered_indices = np.argsort(min_y_coords)
        ordered_cracks = crack_list[ordered_indices]

        if delimiter:
            # Add delimiter cracks at the top and bottom borders
            start_delimiter = np.array([[[0.0, 0.0], [image_height, 0.0]]])
            end_delimiter = np.array([[[0.0, image_width], [image_height, image_width]]])
            ordered_cracks = np.vstack((start_delimiter, ordered_cracks, end_delimiter))

        return ordered_cracks

    def crack_grouping(
        self, 
        ordered_cracks: np.ndarray, 
        threshold: float = 5.0, 
        generate_vertical_crack: bool = True,
        group_within_crack_width: bool = True
    ) -> np.ndarray:
        """
        Group cracks that are close to each other based on endpoint proximity.

        Args:
            ordered_cracks: NumPy array of shape (N, 2, 2) representing ordered cracks.
            threshold: Maximum distance (in pixels) between crack endpoints to consider them close.
            generate_vertical_crack: If True, generates a new vertical crack between close cracks.

        Returns:
            NumPy array of cracks after grouping.
        """
        if len(ordered_cracks) == 0:
            return ordered_cracks

        if group_within_crack_width:
            crack_width = self.avg_crack_width_px*2
            grouped_by_width = []
            i = 0
            while i < len(ordered_cracks):
                group = [ordered_cracks[i]]
                y_ref = np.mean(ordered_cracks[i][:, 1])

                j = i + 1
                while j < len(ordered_cracks):
                    y_next = np.mean(ordered_cracks[j][:, 1])
                    if abs(y_next - y_ref) <= crack_width:
                        group.append(ordered_cracks[j])
                        j += 1
                    else:
                        break

                if len(group) > 1:
                    all_x = [pt[0] for crack in group for pt in crack]
                    min_x, max_x = min(all_x), max(all_x)
                    y_mid = np.mean([np.mean(crack[:, 1]) for crack in group])
                    grouped_by_width.append([[min_x, y_mid], [max_x, y_mid]])
                else:
                    grouped_by_width.append(group[0])

                i = j

            ordered_cracks = np.array(grouped_by_width)

        ordered_cracks = np.array(grouped_by_width)
        updated_cracks = []
        i = 0
        while i < len(ordered_cracks) - 1:
            start_i, end_i = ordered_cracks[i]
            start_next, end_next = ordered_cracks[i + 1]

            # Compute distances between endpoints
            dist_end_i_start_next = np.linalg.norm(end_i - start_next)
            dist_end_next_start_i = np.linalg.norm(end_next - start_i)

            # Choose the closest pair
            if dist_end_i_start_next < dist_end_next_start_i:
                min_dist = dist_end_i_start_next
                group_type = 'end_i_start_next'
            else:
                min_dist = dist_end_next_start_i
                group_type = 'end_next_start_i'

            if min_dist <= threshold:
                if generate_vertical_crack:
                    # Create a new vertical crack between the grouped cracks
                    min_x = min(start_i[0], start_next[0], end_i[0], end_next[0])
                    max_x = max(start_i[0], start_next[0], end_i[0], end_next[0])
                    min_y = min(start_i[1], start_next[1], end_i[1], end_next[1])
                    max_y = max(start_i[1], start_next[1], end_i[1], end_next[1])
                    y_mid = (max_y + min_y) / 2
                    updated_cracks.append([[min_x, y_mid], [max_x, y_mid]])
                else:
                    if group_type == 'end_i_start_next':
                        updated_cracks.append([start_next, end_i])
                    elif group_type == 'end_next_start_i':
                        updated_cracks.append([start_i, end_next])
                    else:
                        raise ValueError(f"Could not group crack number {i}.")
                i += 2  # Skip the next crack as it's already merged
            else:
                updated_cracks.append([start_i, end_i])
                i += 1

        # Add the last crack if not merged
        if i == len(ordered_cracks) - 1:
            updated_cracks.append(ordered_cracks[-1])

        return np.array(updated_cracks)

    def crack_filter(
        self, 
        crack_list: List[np.ndarray], 
        length_threshold: float
    ) -> List[np.ndarray]:
        """
        Filter cracks based on their length.

        Args:
            crack_list: List of cracks for a given frame.
            length_threshold: Minimum length threshold for a crack to be retained.

        Returns:
            Filtered list of cracks.
        """
        filtered_cracks = [
            crack for crack in crack_list
            if crack_length(crack) >= length_threshold
        ]
        return filtered_cracks

    def compute_crack_spacing(
        self, 
        crack_list: List[np.ndarray], 
        plot_dis: bool = False,
    ) -> Tuple[List[float], float, float]:
        """
        Compute the vertical spacing between cracks in a given frame.

        Args:
            crack_list: List of cracks for a given frame.
            plot_dis: Whether to plot the distribution of crack midpoints.

        Returns:
            Tuple containing:
                - List of crack spacings,
                - Average spacing,
                - Standard deviation of spacing.
        """
        if len(crack_list) == 0:
            return [], 0.0, 0.0

        mid_points = [crack_mid_point(crack) for crack in crack_list]
        sorted_mid_points = sorted(mid_points, key=lambda point: point[1])

        crack_spacing = [
            sorted_mid_points[i + 1][1] - sorted_mid_points[i][1]
            for i in range(len(sorted_mid_points) - 1)
        ]

        avg_crack_spacing = np.mean(crack_spacing) if crack_spacing else 0.0
        std_crack_spacing = np.std(crack_spacing) if crack_spacing else 0.0

        return crack_spacing, float(avg_crack_spacing), float(std_crack_spacing)

    def crack_filtering_postprocessing(
        self, 
        cracks: List[np.ndarray],
        avg_crack_grouping_th_px: float = 10.0, 
        crack_length_th: float = 5.0,
        export_images: bool = False,
        background: bool = False,
        remove_outliers: bool = True,
        grouping: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Post-process detected cracks: order, group, filter, and analyze spacing.
        In this subroutine, cracks which follow the following criteria are grouped:
            - Cracks which are distanced less than the defined `self.avg_crack_width_px`
            - Opposing cracks with tips "almost" touching where the max distance is defined by `avg_crack_grouping_th_px`

        Args:
            cracks: List of detected cracks for each frame.
            avg_crack_grouping_th_px: Threshold for grouping cracks (in pixels).
            crack_length_th: Minimum crack length threshold (in pixels).

        Returns:
            List of dictionaries with analysis results for each frame.
        """

        if not cracks:
            return []

        image = imread(self.path_middle_list[0])
        image_height, image_width = image.shape[:2]
        data = []
        cracks_filtered_all = []

        for idx, crack_frame in enumerate(cracks):

            # Performs ordering, grouping, filtering and spacing analysis
            cracks_ordered = self.order_cracks(crack_frame, delimiter=True, image_height=image_height, image_width=image_width)
            cracks_filtered = self.crack_filter(list(cracks_ordered), length_threshold=crack_length_th)
            
            if grouping:
                cracks_grouped = self.crack_grouping(cracks_filtered, threshold=avg_crack_grouping_th_px, generate_vertical_crack=True)
            else:
                cracks_grouped = cracks_filtered

            crack_spacing, avg_crack_spacing, std_crack_spacing = self.compute_crack_spacing(cracks_grouped)

            # print('Number of cracks detected:', len(cracks_filtered)) # For Debugging

            if export_images:
                # Create main subfolder for filtered cracks
                filtered_cracks_subfolder = DataProcessor.generate_folder("Crack Detection", "Filtered Cracks")
                # Create a subfolder with the specimen name inside the filtered cracks folder
                specimen_subfolder = DataProcessor.generate_folder(filtered_cracks_subfolder, self.name)
                
                
                fig, ax = self.plot_cracks(self.image_stack_middle[idx], cracks_filtered, color='black', background_flag=background)
                ax.set_xlabel('x [Px]')
                ax.set_ylabel('y [Px]')
                filename = str(Path(specimen_subfolder) / f"Cracks_{idx}.png")
                fig.savefig(filename)
                plt.close(fig)

            # Remove outliers from crack_spacing 
            if remove_outliers:
                if crack_spacing:
                    crack_spacing_array = np.array(crack_spacing)
                    # Use the interquartile range (IQR) method to detect outliers
                    q1 = np.percentile(crack_spacing_array, 25)
                    q3 = np.percentile(crack_spacing_array, 75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    # Filter out outliers
                    filtered_crack_spacing = crack_spacing_array[
                        (crack_spacing_array >= lower_bound) & (crack_spacing_array <= upper_bound)
                    ]
                    avg_crack_spacing = np.mean(filtered_crack_spacing) if len(filtered_crack_spacing) > 0 else 0.0
                    std_crack_spacing = np.std(filtered_crack_spacing) if len(filtered_crack_spacing) > 0 else 0.0
                else:
                    filtered_crack_spacing = []
                    avg_crack_spacing = 0.0
                    std_crack_spacing = 0.0
            
            data.append({
                'Picture': idx,
                'Avg_spacing': avg_crack_spacing/self.scale_px_mm,
                'Std_spacing': std_crack_spacing/self.scale_px_mm
            })
            
            cracks_filtered_all.append(cracks_filtered)
        
        return data, cracks_filtered_all
    
    def pixels_to_length(self, input_data):
        """
        Convert pixel values to length using the scale factor.
        Accepts either a list of rho values or the output from crack_filtering_postprocessing.

        Args:
            input_data: List of rho values (List[float]) or output from crack_filtering_postprocessing.

        Returns:
            If input is a list of floats: List of scaled values (in mm).
            If input is from crack_filtering_postprocessing: Processed data with scaled values.
        """
        # If input_data is a tuple/list from crack_filtering_postprocessing
        # It typically returns (processed_data, filtered_cracks)
        if isinstance(input_data, (list, tuple)) and len(input_data) == 2:
            processed_data = input_data[0]  # This should be the list of dicts
            
            # Scale the processed data
            scaled = []
            for entry in processed_data:
                avg_spacing = entry.get('Avg_spacing', 0.0) * self.scale_px_mm  # Multiply to convert px to mm
                std_spacing = entry.get('Std_spacing', 0.0) * self.scale_px_mm
                scaled.append({
                    'Picture': entry.get('Picture', None),
                    'Avg_spacing': avg_spacing,
                    'Std_spacing': std_spacing
                })
            return scaled
        
        # If input_data is a list of floats (rho values)
        elif isinstance(input_data, list) and all(isinstance(r, (int, float)) for r in input_data):
            return [r * self.scale_px_mm for r in input_data]

        # If input_data is a list of dicts (already extracted processed data)
        elif isinstance(input_data, list) and all(isinstance(d, dict) for d in input_data):
            scaled = []
            for entry in input_data:
                avg_spacing = entry.get('Avg_spacing', 0.0) * self.scale_px_mm  # Multiply to convert px to mm
                std_spacing = entry.get('Std_spacing', 0.0) * self.scale_px_mm
                scaled.append({
                    'Picture': entry.get('Picture', None),
                    'Avg_spacing': avg_spacing,
                    'Std_spacing': std_spacing
                })
            return scaled
        
        else:
            raise ValueError(f"Input must be a list of floats (rho) or output from crack_filtering_postprocessing. Got: {type(input_data)}")

    def upload_experimental_data(
        self,
        data_path: str,
        sheet_name: Optional[str] = None,
        n0: int = 0,
        nf: Optional[int] = None,
        nstep: int = 1,
    ) -> pd.DataFrame:
        """
        Load and process experimental data from a CSV or Excel file.

        This method loads experimental data containing at least a 'strain_y' column,
        applies optional row filtering, and stores the result in self.experimental_data.
        Supports both CSV and Excel formats.

        Args:
            data_path: Path to the experimental data file (.csv or .xlsx).
            sheet_name: Name of the Excel sheet (only used for Excel files; default: None, uses first sheet).
            n0: Start index for filtering rows (default: 0).
            nf: End index for filtering rows (default: None, meaning till end).
            nstep: Step for row filtering (default: 1).

        Returns:
            pd.DataFrame: Filtered DataFrame containing the 'strain_y' column.
        """
        if data_path.endswith('.csv'):
            # Read CSV file
            df = pd.read_csv(data_path)
        else:
            # Read Excel file
            df = pd.read_excel(data_path, sheet_name=sheet_name)
        df_filtered = df.loc[n0:nf:nstep, ['strain_y']].reset_index(drop=True)
        self.experimental_data = df_filtered

        return df_filtered


    def save_cracks(self, 
                    cracks: List[np.ndarray],
                    folder_name: str = "Crack Detection", 
                    file_name: Optional[str] = None):
        """
        Save the current specimen's cracks data to a pickle file.

        Args:
            folder_name: Directory to save the file.
            file_name: Name of the file (default: uses specimen name).
        """
        
        if file_name is None:
            file_name = f"{self.name}_cracks_data.pkl"
        else:
            # Remove leading underscores or dashes for cleanliness
            file_name = f"{file_name.lstrip('_-')}"
        Path(folder_name).mkdir(parents=True, exist_ok=True)
        file_path = str(Path(folder_name) / file_name)

        with open(file_path, 'wb') as f:
            pickle.dump(cracks, f)
        print(f"Specimen data saved to file: {file_path}")
    
    @staticmethod
    def load_cracks(file_path: str):
        """
        Load cracks data from a pickle file.

        Args:
            file_path: Path to the pickle file.

        Returns:
            Loaded cracks data.
        """
        with open(file_path, 'rb') as f:
            cracks = pickle.load(f)
        return cracks
    
    @staticmethod
    def join_cracks(*crack_lists: List[np.ndarray]) -> List[np.ndarray]:
        """
        Join multiple lists of cracks frame by frame using np.vstack.

        Args:
            *crack_lists: Variable number of lists, each containing arrays of cracks per frame.

        Returns:
            List of joined cracks per frame.
        """
        if not crack_lists:
            return []

        num_frames = len(crack_lists[0])
        for cl in crack_lists:
            if len(cl) != num_frames:
                raise ValueError("All crack lists must have the same number of frames.")

        joined_cracks = []
        for i in range(num_frames):
            cracks_to_join = [cl[i] for cl in crack_lists if cl[i] is not None and len(cl[i]) > 0]

            if cracks_to_join:
                try:
                    joined = np.vstack(cracks_to_join)
                except ValueError as e:
                    raise ValueError(f"vstack failed at frame {i} due to inconsistent shapes: {e}")
            else:
                joined = np.empty((0, 2, 2))  

            joined_cracks.append(joined)

        return joined_cracks


    def full_specimen_eval_transverse(self):
        """
        Utility function that performs transverse cracking for the original (cut) image in a 
        cross-ply specimen.
        
        Used for visualization purpose, so that cracks can be overlayed on the original image but
        not used for any evaluation purposes.
        """
        cracks_transv, _ , _ = self.crack_eval(theta_fd=0, image_stack_orig=True)
        cracks_splitting, _ , _ = self.crack_eval(theta_fd=90, image_stack_orig=True)
        return cracks_transv, cracks_splitting


    def crack_eval_crossply(
        self,
        export_images: bool = False, 
        background: bool = False, 
        comparison: bool = False,
        post_processing: bool = False,
        avg_crack_grouping_th_px: float = 50.0,
        save_cracks: bool = False,
        color_cracks: str = 'red',
        timing: bool = True,
        output_dir: str = None,
        c_length_th: Optional[float] = None
    ) -> Tuple[List[np.ndarray], List[float], List[float], List[np.ndarray], List[float], List[float]]:
        """
        Special case, for crack evaluation for cross-ply laminates, where theta_fd = 0 and 90 degrees.

        Args:
            theta_fd: Crack orientation angle.
            export_images: Whether to export identified crack images.
            background: Whether to include background in the plots.
            comparison: Whether to include comparison in the plots.
            timing: Whether to print elapsed time for the analysis.
            output_dir: Optional base directory for saving results. If None, uses current working directory.
            c_length_th: Optional crack length threshold for filtering. 
                If None, defaults to 0.20 * width * scale_px_mm.

        Returns:
            A tuple containing cracks, rho, and theta values. Results in mm if post_processing is True.
        """

        # Default crack length threshold for filtering
        if c_length_th is None:
            c_length_th = 0.20 * self.dimensions['width'] * self.scale_px_mm


        start_time = time.time()
        
        # Determine the base directory for saving cracks
        if output_dir is not None:
            base_dir = str(Path(output_dir).resolve())
        else:
            base_dir = str(Path.cwd())
        self.identified_cracks_subfolder = str(Path(base_dir) / "Crack Detection" / "Identified Cracks")
        sample_subfolder = str(Path(self.identified_cracks_subfolder) / f"{self.name}__crossply")
        
        file_90 = str(Path(sample_subfolder) / f"{self.name}_cracks_data_90.pkl")
        file_0 = str(Path(sample_subfolder) / f"{self.name}_cracks_data_0.pkl")
        cracks_90 = cracks_0 = rho_90 = th_90 = rho_0 = th_0 = None
        # Check if cracks are already saved
        if Path(file_90).exists() and Path(file_0).exists():
            print(f"[{self.name}] Found saved cracks. Skipping evaluation.")
            with open(file_90, 'rb') as f:
                cracks_90 = pickle.load(f)
            with open(file_0, 'rb') as f:
                cracks_0 = pickle.load(f)

            rho_90, th_90, rho_0, th_0 = [], [], [], []
        else:
            cracks_90, rho_90, th_90 = self.crack_eval(theta_fd=0, 
                                                    background=background, 
                                                    comparison=comparison,
                                                    color_cracks=color_cracks)

            cracks_0, rho_0, th_0 = self.crack_eval(theta_fd=90, 
                                                background=background, 
                                                comparison=comparison,
                                                color_cracks=color_cracks)

            # Save cracks if requested
            if save_cracks:
                # Ensure the parent directory for the files exists
                parent_dir = Path(file_90).parent
                parent_dir.mkdir(parents=True, exist_ok=True)
                with open(file_90, 'wb') as f:
                    pickle.dump(cracks_90, f)
                with open(file_0, 'wb') as f:
                    pickle.dump(cracks_0, f)

        if export_images and post_processing == False:
                    
            sample_subfolder = DataProcessor.generate_folder(self.identified_cracks_subfolder, f"{self.name}__crossply")
             
            for idx, _ in enumerate(cracks_90):
                if cracks_90[idx] is not None and len(cracks_90[idx]) > 0 and cracks_0[idx] is not None and len(cracks_0[idx]) > 0:
                    combined_cracks = np.vstack((cracks_90[idx], cracks_0[idx]))
                else:
                    combined_cracks = []
                fig, ax = self.plot_cracks(self.image_stack_middle[idx], combined_cracks, 
                                           background_flag = background, 
                                           color=color_cracks,
                                           comparison=comparison)
                ax.set_xlabel('x [Px]')
                ax.set_ylabel('y [Px]')
                filename = str(Path(sample_subfolder) / f"Cracks_{idx}.png")
                fig.savefig(filename)
                plt.close(fig)

        if post_processing:
            processed_data = self.crack_filtering_postprocessing(cracks = cracks_90, 
                                                avg_crack_grouping_th_px=avg_crack_grouping_th_px,
                                                crack_length_th=c_length_th,
                                                export_images=False,
                                                background=background,
                                                remove_outliers=True)
            
            if export_images:

                sample_subfolder = DataProcessor.generate_folder(self.identified_cracks_subfolder, f"{self.name}__crossply")
            
                for idx, _ in enumerate(cracks_90):
                    if processed_data[idx]['Cracks_filtered'] is not None and len(processed_data[idx]['Cracks_filtered']) > 0 and cracks_0[idx] is not None and len(cracks_0[idx]) > 0:
                        combined_cracks = np.vstack((processed_data[idx]['Cracks_filtered'], cracks_0[idx]))
                    else:
                        combined_cracks = []
                    fig, ax = self.plot_cracks(self.image_stack_middle[idx], combined_cracks, 
                                            background_flag = background, 
                                            color=color_cracks,
                                            comparison=comparison)
                    ax.set_xlabel('x [Px]')
                    ax.set_ylabel('y [Px]')
                    filename = str(Path(sample_subfolder) / f"Cracks_{idx}.png")
                    fig.savefig(filename)
                    plt.close(fig)
            
            
            results_mm = self.pixels_to_length(processed_data)
            self.export_crack_spacing(results_mm, output_dir, f"{self.name}_crack_spacing_data.csv")
    
        if save_cracks:
            DataProcessor.save_cracks_to_file(cracks_90, sample_subfolder, f"{self.name}_cracks_data_90.pkl")
            DataProcessor.save_cracks_to_file(cracks_0, sample_subfolder, f"{self.name}_cracks_data_0.pkl") 
               
        elapsed = time.time() - start_time
        if timing:
            print(f"[{self.name}] crack_eval_crossply completed in {elapsed:.2f} seconds.")
        return cracks_90, rho_90, th_90, cracks_0, rho_0, th_0
    

    def crack_eval_plus_minus(
        self,
        theta_fd: int,
        export_images: bool = False, 
        background: bool = False, 
        comparison: bool = False,
        transverse_layer: bool = False,
        post_processing: bool = False,
        avg_crack_grouping_th_px: float = 50.0,
        save_cracks: bool = False,
        color_cracks: str = 'red',
        color_transverse: str = 'blue',
        timing: bool = True,
        output_dir: str = None,
        c_length_th: Optional[float] = None
    ) -> Tuple[List[np.ndarray], List[float], List[float], List[np.ndarray], List[float], List[float]]:
        """
        Special case, for crack evaluation for plus and minus laminates, where theta_fd = +/- theta. 
        Additionally, it can also be used if a transverse layer is included in the laminate stacking sequence.

        Args:
            theta_fd: Crack orientation angle for the +/- theta laminate.
            export_images: Whether to export identified crack images.
            background: Whether to include background in the plots.
            comparison: Whether to include comparison in the plots.
            timing: Whether to print elapsed time for the analysis.
            output_dir: Optional base directory for saving results. If None, uses current working directory.
            c_length_th: Optional crack length threshold for filtering. 
                If None, defaults to 0.20 * width * scale_px_mm.

        Returns:
            A tuple containing cracks, rho, and theta values. In mm.
        """

        if c_length_th is None:
            c_length_th = 0.20 * self.dimensions['width'] * self.scale_px_mm

        start_time = time.time()

        # Setup output folders
        if output_dir is not None:
            base_dir = str(Path(output_dir).resolve())
        else:
            base_dir = str(Path.cwd())
        self.identified_cracks_subfolder = str(Path(base_dir) / "Crack Detection" / "Identified Cracks")
        sample_subfolder = str(Path(self.identified_cracks_subfolder) / f"{self.name}__plus_minus_{theta_fd}")

        file_90 = str(Path(sample_subfolder) / f"{self.name}_cracks_data_90.pkl")
        file_theta_plus = str(Path(sample_subfolder) / f"{self.name}_cracks_data_plus_{theta_fd}.pkl")
        file_theta_minus = str(Path(sample_subfolder) / f"{self.name}_cracks_data_minus_{theta_fd}.pkl")

        # Try to load cracks if already saved
        cracks_plus = cracks_minus = rho_plus = th_plus = rho_minus = th_minus = None
        cracks_90 = rho_90 = th_90 = None
        if Path(file_theta_plus).exists() and Path(file_theta_minus).exists():
            with open(file_theta_plus, 'rb') as f:
                cracks_plus = pickle.load(f)
            with open(file_theta_minus, 'rb') as f:
                cracks_minus = pickle.load(f)
            # Optionally load rho/th if you save them separately
            rho_plus, th_plus, rho_minus, th_minus = [], [], [], []
            if transverse_layer and Path(file_90).exists():
                with open(file_90, 'rb') as f:
                    cracks_90 = pickle.load(f)
                rho_90, th_90 = [], []
        else:
            cracks_plus, rho_plus, th_plus = self.crack_eval(theta_fd=+theta_fd, 
                                                    background=background, 
                                                    comparison=comparison,
                                                    color_cracks=color_cracks)

            cracks_minus, rho_minus, th_minus = self.crack_eval(theta_fd=-theta_fd, 
                                                background=background, 
                                                comparison=comparison,
                                                color_cracks=color_cracks)

            if transverse_layer:
                cracks_90, rho_90, th_90 = self.crack_eval(theta_fd=0, 
                                                        background=background, 
                                                        comparison=comparison,
                                                        color_cracks=color_cracks)

            if save_cracks:
                Path(sample_subfolder).mkdir(parents=True, exist_ok=True)
                with open(file_theta_plus, 'wb') as f:
                    pickle.dump(cracks_plus, f)
                with open(file_theta_minus, 'wb') as f:
                    pickle.dump(cracks_minus, f)
                if transverse_layer:
                    with open(file_90, 'wb') as f:
                        pickle.dump(cracks_90, f)

        # Export images
        if export_images and not post_processing:
            sample_subfolder = DataProcessor.generate_folder(self.identified_cracks_subfolder, f"{self.name}__plus_minus_{theta_fd}")
            for idx in range(len(cracks_plus)):
                combined_cracks = []
                if cracks_plus[idx] is not None and len(cracks_plus[idx]) > 0:
                    combined_cracks.append(cracks_plus[idx])
                if cracks_minus[idx] is not None and len(cracks_minus[idx]) > 0:
                    combined_cracks.append(cracks_minus[idx])
                if combined_cracks:
                    combined = np.vstack(combined_cracks)
                else:
                    combined = np.empty((0, 2, 2))

                if transverse_layer and cracks_90 is not None and cracks_90[idx] is not None and len(cracks_90[idx]) > 0:
                    fig, ax = self.plot_cracks(
                        self.image_stack_middle[idx], cracks_90[idx],
                        background_flag=background,
                        color=color_transverse,
                        comparison=comparison
                    )
                    if combined is not None and len(combined) > 0:
                        self.plot_cracks(
                            self.image_stack_middle[idx], combined,
                            ax=ax,
                            background_flag=False,
                            color=color_cracks,
                            comparison=comparison
                        )
                else:
                    fig, ax = self.plot_cracks(
                        self.image_stack_middle[idx], combined,
                        background_flag=background,
                        color=color_cracks,
                        comparison=comparison
                    )
                ax.set_xlabel('x [Px]')
                ax.set_ylabel('y [Px]')
                filename = str(Path(sample_subfolder) / f"Cracks_{idx}.png")
                fig.savefig(filename)
                plt.close(fig)

        
        if post_processing and transverse_layer:
            processed_data = self.crack_filtering_postprocessing(
                cracks=cracks_90,
                avg_crack_grouping_th_px=avg_crack_grouping_th_px,
                crack_length_th=c_length_th,
                export_images=False,
                background=background,
                remove_outliers=True
            )

            results_mm = self.pixels_to_length(processed_data)
            self.export_crack_spacing(results_mm, output_dir, f"{self.name}_crack_spacing_data.csv")
        elif post_processing and not transverse_layer:
            print("Post-processing is only available when a transverse layer is included. Skipping post-processing.")

        if save_cracks:
            DataProcessor.save_cracks_to_file(cracks_plus, sample_subfolder, f"{self.name}_cracks_data_plus_{theta_fd}.pkl")
            DataProcessor.save_cracks_to_file(cracks_minus, sample_subfolder, f"{self.name}_cracks_data_minus_{theta_fd}.pkl")
            if transverse_layer:
                DataProcessor.save_cracks_to_file(cracks_90, sample_subfolder, f"{self.name}_cracks_data_90.pkl")

        elapsed = time.time() - start_time
        if timing:
            print(f"[{self.name}] crack_eval_plus_minus completed in {elapsed:.2f} seconds.")

        if transverse_layer:
            return cracks_plus, rho_plus, th_plus, cracks_minus, rho_minus, th_minus, cracks_90, rho_90, th_90
        else:
            return cracks_plus, rho_plus, th_plus, cracks_minus, rho_minus, th_minus
