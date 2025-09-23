# Shift Correction Code

## 1. Introduction

Here, one of the auxiliary scripts which support DelaDect can be found. In this script, a shift-distortion correction procedure is provided. Having a series of shift-corrected images is required for proper crack and delamination detection. More about shift correction can be found [here](https://crackdect.readthedocs.io/en/latest/shift_correction.html).

Inside the [CrackDect](https://github.com/mattdrvo/CrackDect) package, a shift-distortion correction procedure was provided; however, it had some limitations for static testing and was deemed unreliable for our test results. However, it may be sufficient for the user's experimental data.

In the present script, a shift-correction algorithm is provided with a simple Graphical User Interface (GUI) which can be used as auxiliary to the DelaDect tool.

## 2. Requirements

Below, the required Python libraries are described:

### Required Libraries:
- [matplotlib](https://pypi.org/project/matplotlib/) `3.7.2`
- [numpy](https://pypi.org/project/numpy/) `1.24.3`
- [Pillow](https://pypi.org/project/pillow/) `9.4.0`
- [scipy](https://pypi.org/project/scipy/) `1.11.1`
- [skimage](https://scikit-image.org/)

### Additional Requirements:
- Figures need to be already cut, and the markers need to be visible.
- All the figures must be in the same folder.

## 3. How to Use

1. **Prepare the Images**  
    Ensure all the pictures are in a single folder:  
    ![](media/images_folder.png)

2. **Run the Script**  
    In the folder where the script is located, open the `cmd` or `Anaconda cmd`. Make sure you are in the same folder. You can use the command:

    ```bash
    cd "Z:\path"
    ```

    to navigate to the folder using the console. Then, run the Python script by:

    ```bash
    python shift_correction.py
    ```

    Alternatively, you can also run the Python script from any IDE.

3. **Open the First Image**  
    With the Shift Correction GUI opened, go to `File -> Open First Image` and select the first image of the series:  
    ![](media/app.png)

4. **Set the Output Folder**  
    Select where you want to save the shift-corrected images by going to `File -> Save Images In`. Choose a folder.

5. **Mark the Points**  
    Click on the points of the markers using the GUI by pressing `Ctrl + Left Click`. If you make a mistake, you can delete the points by using `Shift + Left Click`. Try to add points in the center of each marker:  
    ![](media/selection.png)

6. **Perform Shift Correction**  
    Finally, go to `File -> Perform Shift Correction`. Check the console to monitor the progress.

## 4. Commands

The commands used depend on the operating system, but most commands are universal. Below is a summary of the key actions:

**For Windows/Linux:**
- **Add a point:** `Ctrl + Left Click`

**For macOS:**
- **Add a point:** `Command + Left Click`

**Common Commands:**
- **Pan the figure:** `Left Click`
- **Zoom in or out:** `Mouse Wheel`
- **Delete a point:** `Shift + Left Click`

## 5. Outputs
The output of this program is obviously the shift corrected images, but it is also possible to see the tracking of the points to see if the program is properly tracking the correct points on each image.

![](media/0105_dic_test.png)

## 6. Settings
In the GUI, there are additional settings that can be adjusted to fine-tune the shift correction process:

- **Step Size (`n`)**:  
    This determines the number of images to skip during evaluation. For example:
    - If `n = 1`, all images are processed.
    - If `n = 2`, only every second image is considered for shift correction, and so on.

- **Threshold Value**:  
    This sets the maximum pixel intensity considered as "black" in the grayscale image. It plays a crucial role in marker detection:
    - Lower values focus on darker pixels, but if the threshold is too low, detection might fail.
    - Higher values include more pixels, but if set too high, too many points may be detected, leading to inaccuracies.  
    A good starting point for this value is between **30-50** in grayscale, but keep in mind that lighting conditions can significantly impact this parameter.

- **Gaussian Filter**:  
    This filter is useful when markers are poorly defined, and multiple points are detected for each marker. Applying a Gaussian filter smooths the image, improving detection. A recommended range for this setting is **1-5**.

- **Median Filter**:  
    This filter helps when there are inconsistencies within the marker. It averages deviations inside the marker, enhancing point detection accuracy.

The suggested values above work for the provided example images, however they should be adapted accordingly to the user images.
