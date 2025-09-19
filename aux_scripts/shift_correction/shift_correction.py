"""
Module: shift_correction.py

This module provides classes and functions for performing shift correction and strain evaluation on images of specimens. 
It includes a graphical user interface (GUI) for selecting points on the images and setting parameters for the analysis.
Refer to the README.md file for more instructions.

Feel free to provide any contribution or improvement to this code by sending us an email.

Authors:
    Vasco D.C Pires
    Matthias Rettl
Affiliation:
    Chair of Designing Plastics and Composite Materials
    University of Leoben
    URL: https://www.kunststofftechnik.at/en/konstruieren
Classes:
    SpecimenVideo
        A class to handle loading and accessing images from a specified directory.
    DIC
        A class to perform Digital Image Correlation (DIC) for strain evaluation and shift correction.
    ImagePointExtractor
        A class to provide a GUI for selecting points on images and setting parameters for DIC analysis.
"""



#Required Packages
import tkinter as tk
from tkinter import filedialog, simpledialog
from PIL import Image, ImageTk
import platform

import csv
import os
from functools import lru_cache
import numpy as np
from skimage import io, registration, morphology
from skimage import feature
from scipy import ndimage, interpolate, spatial, sparse
from matplotlib import pyplot as plt
import shutil


################################

ending_suffix = "cycles.bmp"        # The ending suffix of the images to be processed

# Files types supported: .jpg, .jpeg, .bmp

#################################


#Initializer 
ACTIONS = {
    "shift_correction": False,
    "strain_evaluation": False,
    "both": False
}

class SpecimenVideo:
    def __init__(self, PATH):
        """
        SpecimenVideo class initializer

        Parameters
        ----------
        PATH : str
            The path to the folder containing the images to be processed

        Returns
        -------
        None
        """
        self.image_paths = [
            f"{PATH}/{file_name}"
            for file_name in sorted(
                [
                    file_name
                    for file_name in os.listdir(PATH)
                    if os.path.isfile(f"{PATH}/{file_name}") and file_name.endswith(ending_suffix)
                ],
                key=lambda file_name: int(file_name.removesuffix(ending_suffix))
            )
        ]

    @lru_cache(1)
    def get_image(self, frame):
        return io.imread(fname=self.image_paths[frame], as_gray=True)


class DIC:
    MAX_DISTANCE = 20   # Max distance in px from the detected point and previous

    def __init__(self, initial_points, median_filter_flag = True, treshold_value = 40, 
                 gaussian_filter_value = 1, specimen_name = 'default', 
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
            The value below which the pixels are considered to be part of the pattern. Defaults to 40.
        gaussian_filter_value : int, optional
            The sigma value of the gaussian filter to use to pre-process the images. Defaults to 1.
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
            ref_tree.sparse_distance_matrix(tree, max_distance=self.MAX_DISTANCE, p=2)
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

    def start(self, image: np.ndarray, plot: bool = False, initial_image_path = None):
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
        print(self.list_points)
        self.list_interpolator = [self.interpolator(self.list_points[0], self.list_points[-1])]
        
        
        if ACTIONS["shift_correction"]:
            # Construct the destination path with the new name
            save_dir = os.path.join(self.output_path, self.specimen_name, "shift_corrected")
            os.makedirs(save_dir, exist_ok=True)
                    
            # Get the list of files in the source folder
            files = os.listdir(self.image_path)

            # Filter out only image files (optional, based on your image formats)
            image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
            
            if image_files:
                # Get the first image file
                first_image = image_files[0]
                
                # Define full paths
                source_image_path = initial_image_path
                destination_image_path = os.path.join(save_dir, '0001_sc.bmp')
                
                # Log the source and destination paths
                print(f"Source image path: {source_image_path}")
                print(f"Destination image path: {destination_image_path}")
                
                # Copy the image
                shutil.copy2(source_image_path, destination_image_path)
                print(f"Copied {first_image} to {destination_image_path}")
            else:
                print("No image found in the source folder.")
                
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

            fig.savefig(f"tracking_{len(self.list_points):04d}.png", dpi=300)
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
        
        image_0 = self.list_images[0]
        image_idx = self.list_images[idx]
        image_idx_ = np.zeros(image_idx.shape + np.array([2*d, 2*d]), dtype=image_idx.dtype)
        image_idx_[d:-d, d:-d] = image_idx

        points_0 = self.list_points[0]
        points_idx = np.copy(self.list_points[idx])

        mask = np.all(np.isfinite(points_idx), axis=1)
        for i in range(len(points_idx)):
            if mask[i]:
                point_0 = points_0[i].astype(int)
                point_idx = points_idx[i].astype(int)

                shift, error, *_ = registration.phase_cross_correlation(
                    image_0[point_0[0] - d:point_0[0] + d, point_0[1] - d:point_0[1] + d],
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

        WARNING: This function is very simplistic and only works currently for 8 points,
                 as seen in the provided example image. The results obtained from this 
                 function should be adapted to the specific use case and compared with 
                 other strain measuring methods.
        
        The order of the points should be given as the following in counter-clockwise order:
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
            
            initial_len_y = [np.abs(initial_points[0][1] - initial_points[-1][1]),
                            np.abs(initial_points[2][1] - initial_points[3][1]),
                            np.abs(initial_points[3][1] - initial_points[4][1]),
                            np.abs(initial_points[6][1] - initial_points[7][1])]

            points_idx = self.list_points[idx]

            current_len_y = [np.abs(points_idx[0][1] - points_idx[-1][1]),
                            np.abs(points_idx[2][1] - points_idx[3][1]),
                            np.abs(points_idx[3][1] - points_idx[4][1]),
                            np.abs(points_idx[6][1] - points_idx[7][1])]

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

            initial_len_x = [np.abs(initial_points[0][0] - initial_points[1][0]),
                             np.abs(initial_points[2][0] - initial_points[1][0]),
                             np.abs(initial_points[4][0] - initial_points[5][0]),
                             np.abs(initial_points[5][0] - initial_points[6][0])]


            current_len_x = [np.abs(points_idx[0][0] - points_idx[1][0]),
                             np.abs(points_idx[2][0] - points_idx[1][0]),
                             np.abs(points_idx[4][0] - points_idx[5][0]),
                             np.abs(points_idx[5][0] - points_idx[6][0])]


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


########### GUI ############

"""
Overview:
    The GUI application allows users to:
        - Open an initial image and select points for analysis.
        - Save selected points to a file.
        - Configure settings for image processing and analysis.
        - Perform shift correction and strain evaluation on a sequence of images.
        - Visualize and save the results of the analysis.
"""

class ImagePointExtractor:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Point Extractor")
        
        self.canvas = tk.Canvas(root, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.points = []
        self.zoom_level = 1.0
        self.initial_image_path = None
        self.image = None
        self.saved_path = None
        self.pan_start = None
        self.image_offset = [0, 0]
        
        # Inputs
        self.n = 1
        self.median_filter_flag = False
        self.threshold_value = 40
        self.gaussian_filter_value = 1
        
        self.menu = tk.Menu(root)
        self.root.config(menu=self.menu)
        
        self.file_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Open First Image", command=self.open_image)
        self.file_menu.add_command(label="Save Images In", command=self.save_image_in)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Perform Shift Correction (check terminal)", command=self.perform_eval)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Perform strain evaluation", command=self.perform_strain)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Perform Shift Correction and strain evaluation", command=self.perform_all)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Save points to a .txt file", command=self.save_points)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Settings", command=self.open_settings)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=root.quit)
        
        self.canvas.bind("<Button-1>", self.start_pan)
        self.canvas.bind("<B1-Motion>", self.pan_image)
        self.canvas.bind("<ButtonRelease-1>", self.end_pan)
        

        # Here, added compatibility for macOS is added
        if platform.system() == "Darwin":  # macOS
            self.canvas.bind("<Command-Button-1>", self.add_point)  # Use Command key on macOS
        else:
            self.canvas.bind("<Control-Button-1>", self.add_point) 
        
        
        self.canvas.bind("<Shift-Button-1>", self.delete_point)
        self.canvas.bind("<MouseWheel>", self.zoom)
    
    def start_pan(self, event):
        self.pan_start = (event.x, event.y)
    
    def pan_image(self, event):
        if self.pan_start:
            dx = event.x - self.pan_start[0]
            dy = event.y - self.pan_start[1]
            self.canvas.move(tk.ALL, dx, dy)
            self.image_offset[0] += dx
            self.image_offset[1] += dy
            self.pan_start = (event.x, event.y)
            self.redraw_points()
    
    def end_pan(self, event):
        self.pan_start = None

    def open_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.initial_image_path = file_path
            self.image = Image.open(file_path)
            self.zoom_level = 0.8
            self.display_image()
            self.points = []  # Reset points
            self.saved_path = os.path.dirname(file_path)  # Save the directory of the image file
            print(f"Image folder path: {self.saved_path}")  # Print the saved path for verification

    def save_image_in(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.output_folder = folder_path
            print(f"Output folder selected: {self.output_folder}")
    
    def display_image(self):
        if self.image:
            width, height = self.image.size
            size = int(width * self.zoom_level), int(height * self.zoom_level)
            resized_image = self.image.resize(size, Image.ANTIALIAS)
            self.photo_image = ImageTk.PhotoImage(resized_image)
            self.canvas.delete("all")
            self.canvas.config(width=self.photo_image.width(), height=self.photo_image.height())
            self.canvas.create_image(self.image_offset[0], self.image_offset[1], image=self.photo_image, anchor=tk.NW)
            self.redraw_points()
    
    def add_point(self, event):
        x, y = int((event.x - self.image_offset[0]) / self.zoom_level), int((event.y - self.image_offset[1]) / self.zoom_level)
        self.points.append((y, x))  # Inverting coordinates
        self.redraw_points()
    
    def delete_point(self, event):
        x, y = int((event.x - self.image_offset[0]) / self.zoom_level), int((event.y - self.image_offset[1]) / self.zoom_level)
        delete_range = 5  # Define the range within which a point can be deleted
        for point in self.points:
            if abs(point[0] - y) <= delete_range and abs(point[1] - x) <= delete_range:
                self.points.remove(point)
                break
        self.redraw_points()

    def save_points(self):
        if self.points:
            save_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
            if save_path:
                with open(save_path, 'w') as f:
                    for point in self.points:
                        f.write(f"{point[0]}, {point[1]}\n")
                print(f"Points saved to {save_path}")
        else:
            print("No points to save")
    
    def zoom(self, event):
        if event.delta > 0:
            self.zoom_level *= 1.1
        else:
            self.zoom_level /= 1.1
        self.display_image()
    
    def redraw_points(self):
        self.canvas.delete("points")
        for point in self.points:
            y, x = point  # Since points are stored as (y, x)
            display_x, display_y = int(x * self.zoom_level) + self.image_offset[0], int(y * self.zoom_level) + self.image_offset[1]
            self.canvas.create_oval(display_x-2, display_y-2, display_x+2, display_y+2, fill="red", tags="points")
            self.canvas.create_text(display_x, display_y, text=f'({point[0]},{point[1]})', anchor=tk.NW, fill="red", tags="points")
    
    def open_settings(self):
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        
        tk.Label(settings_window, text="Figure interval n:").grid(row=0, column=0)
        n_entry = tk.Entry(settings_window)
        n_entry.insert(0, str(self.n))
        n_entry.grid(row=0, column=1)
        
        tk.Label(settings_window, text="Median Filter:").grid(row=1, column=0)
        median_filter_var = tk.BooleanVar(value=self.median_filter_flag)
        median_filter_check = tk.Checkbutton(settings_window, variable=median_filter_var)
        median_filter_check.grid(row=1, column=1)
        
        tk.Label(settings_window, text="Threshold Value:").grid(row=2, column=0)
        threshold_entry = tk.Entry(settings_window)
        threshold_entry.insert(0, str(self.threshold_value))
        threshold_entry.grid(row=2, column=1)
        
        tk.Label(settings_window, text="Gaussian Filter Value:").grid(row=3, column=0)
        gaussian_entry = tk.Entry(settings_window)
        gaussian_entry.insert(0, str(self.gaussian_filter_value))
        gaussian_entry.grid(row=3, column=1)
        
        def save_settings():
            self.n = int(n_entry.get())
            self.median_filter_flag = median_filter_var.get()
            self.threshold_value = int(threshold_entry.get())
            self.gaussian_filter_value = int(gaussian_entry.get())
            settings_window.destroy()
        
        save_button = tk.Button(settings_window, text="Save", command=save_settings)
        save_button.grid(row=4, columnspan=2)
    
    def perform_eval(self):
        ACTIONS["shift_correction"] = True
        ACTIONS["strain_evaluation"] = False
        self.root.quit()
    
    def perform_strain(self):
        ACTIONS["shift_correction"] = False
        ACTIONS["strain_evaluation"] = True
        self.root.quit()
    
    def perform_all(self):
        ACTIONS["both"] = True
        self.root.quit()
        
    def correction(self):
        if ACTIONS["both"]:
            ACTIONS["shift_correction"] = True
            ACTIONS["strain_evaluation"] = True
            
        # Performs the correction
        import tracemalloc
    
        tracemalloc.start()
        snapshot = tracemalloc.take_snapshot()

        initial_points = np.array(self.points)    
        
        n = self.n
         
        spec = SpecimenVideo(self.saved_path)
        dic = DIC(initial_points, self.median_filter_flag, self.threshold_value, self.gaussian_filter_value, self.output_folder, self.saved_path)

        dic.start(image=spec.get_image(0), plot=True, initial_image_path = self.initial_image_path)
        
        strain_x_list = []
        strain_y_list = []
        
        for i in range(1, len(spec.image_paths), n):
            print(i)
            j = len(dic.list_images)
            dic.next_image(image=spec.get_image(i))
            dic.fit_matched_points_accurately(j)
            if ACTIONS["shift_correction"]:
                dic.back_transform_image(j, plot=True)
            if ACTIONS["strain_evaluation"]:
                strain_x, strain_y = dic.eval_strain(j)
                strain_x_list.append(strain_x)
                strain_y_list.append(strain_y)
            dic.plot_marker(j)
        
        if ACTIONS["strain_evaluation"]:
            output_path = os.path.join(self.output_folder, 'strain_data.csv')
            with open(output_path, 'w', newline='') as csvfile:
                fieldnames = ['strain_x', 'strain_y']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for x, y in zip(strain_x_list, strain_y_list):
                    writer.writerow({'strain_x': x, 'strain_y': y})
                
            print("Strain Evaluation is done. Check results in:")  
            print(str(output_path))         
        
        #snapshot2 = tracemalloc.take_snapshot()
        #top_stats = snapshot.statistics('traceback')
        
        if ACTIONS["shift_correction"]:
            print("Shift correction is done")

    
              
if __name__ == "__main__":

    
    root = tk.Tk()
    app = ImagePointExtractor(root)
    root.mainloop()
    
    if ACTIONS["shift_correction"] or ACTIONS["strain_evaluation"] or ACTIONS["both"]:
        app.correction()


