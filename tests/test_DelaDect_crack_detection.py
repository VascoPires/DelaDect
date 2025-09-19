import os
import pytest
from deladect import Specimen
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# SciencePlots

# import scienceplots
# plt.style.use(['science', 'grid', 'high-vis'])

# plt.rcParams['figure.dpi'] = 600
# plt.rcParams['savefig.bbox'] = 'tight'

# Test configuration
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAMPLE_DATA_DIR = os.path.join(REPO_ROOT, "example_images", "sample-1")

# Test folders - organized by test type and timestamp
TEST_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
TESTS_BASE_DIR = os.path.join(REPO_ROOT, "tests", "test_results", f"test_run_{TEST_TIMESTAMP}")

# Create specific test directories
TEST_DIRS = {
    'basic_functionality': os.path.join(TESTS_BASE_DIR, "01_basic_functionality"),
    'crossply_analysis': os.path.join(TESTS_BASE_DIR, "02_crossply_analysis"),
    'visualization': os.path.join(TESTS_BASE_DIR, "03_visualization"),
    'postprocessing': os.path.join(TESTS_BASE_DIR, "04_postprocessing")
}

# Sample configuration
specimens = {
    'sample-1': {
        'dimensions': {
            'width': 20.13,
            'thickness': 2.27
        },
        'scale': 41.03328366,
        'theta_in': 0,
        'theta_out': 90,
        'image_paths': {
            'cut': os.path.join(SAMPLE_DATA_DIR, "cut"),
            'upper': os.path.join(SAMPLE_DATA_DIR, "upper"),
            'lower': os.path.join(SAMPLE_DATA_DIR, "lower"),
            'middle': os.path.join(SAMPLE_DATA_DIR, "middle"),
            'experimental_data': os.path.join(SAMPLE_DATA_DIR, "experimental_data.csv")
        }
    }
}

@pytest.fixture(scope="module", autouse=True)
def setup_test_directories():
    """Create all test directories before running tests"""
    for dir_path in TEST_DIRS.values():
        os.makedirs(dir_path, exist_ok=True)
    yield


@pytest.fixture(scope="function")
def specimen_instance():
    sample_name = "sample-1"
    specimen_data = specimens[sample_name]
    
    return Specimen(
        name=sample_name,
        dimensions=specimen_data['dimensions'],
        scale_px_mm=specimen_data['scale'],
        path_cut=specimen_data['image_paths']['cut'],
        path_upper_border=specimen_data['image_paths']['upper'],
        path_lower_border=specimen_data['image_paths']['lower'],
        path_middle=specimen_data['image_paths']['middle'],
        sorting_key='_sc',
        image_types=['png']
    )

def test_basic_crack_detection(specimen_instance):
    test_dir = TEST_DIRS['basic_functionality']
    
    # Test crack evaluation
    cracks, rho, th = specimen_instance.crack_eval(
        theta_fd=specimens['sample-1']['theta_in'],
        background=True,
        export_images=True,
        color_cracks='black',
        comparison=False,
        image_stack_orig=True
    )

    # Convert to physical units
    rho_scaled = specimen_instance.pixels_to_length(rho)
    
    # Export results
    specimen_instance.export_rho_th(rho_scaled, th, test_dir, "crack_density_results.csv")
    
    # Save cracks for later analysis
    specimen_instance.save_cracks(
        cracks=cracks, 
        folder_name=test_dir, 
        file_name="detected_cracks.pkl"
    )

    # Test export results with experimental strain included
    specimen_instance.upload_experimental_data(specimens['sample-1']['image_paths']['experimental_data'])
    specimen_instance.export_rho_th(rho_scaled, th, test_dir, "crack_density_results_with_strain.csv")

    # File paths
    results_file = os.path.join(test_dir, "crack_density_results.csv")
    results_file_strain = os.path.join(test_dir, "crack_density_results_with_strain.csv")

    # Assertions
    assert isinstance(cracks, list), "Cracks should be a list"
    assert isinstance(rho, list), "Rho should be a list"
    assert isinstance(th, list), "Theta should be a list"
    assert len(cracks) > 0, "No cracks detected"
    assert len(rho) == len(th), "Rho and Theta should have same length"
    
    # Verify files were created
    assert os.path.exists(results_file), "Results file should be created"
    assert os.path.exists(results_file_strain), "Results file with strain should be created"
    assert os.path.exists(os.path.join(test_dir, "detected_cracks.pkl")), "Cracks file should be created"

def test_crossply_analysis(specimen_instance):
    """Test for the utility function which handles cross-ply specimens"""
    test_dir = TEST_DIRS['crossply_analysis']
    
    # Perform cross-ply analysis
    cracks_90, rho_90, th_90, cracks_0, rho_0, th_0 = specimen_instance.crack_eval_crossply(
        background=False,
        export_images=True,
        color_cracks='black',
        comparison=False
    )

    # rho90 and rho0 are the crack densities for 90° and 0° cracks respectively provided by CrackDect
    # Which is computed based on the paper by rho = sum(crack_lengths)/specimen_area

    # Post-processing and filtering
    w = specimens['sample-1']['dimensions']['width'] * specimens['sample-1']['scale']   # specimen width in pixels

    processed_data, filtered_cracks = specimen_instance.crack_filtering_postprocessing(
        cracks_90, 120, 0.20*w, remove_outliers=True)

    # Convert to physical units
    results_mm = specimen_instance.pixels_to_length(processed_data)
    
    # Export crack spacing
    spacing_file = os.path.join(test_dir, "crack_spacing_results.csv")
    specimen_instance.export_crack_spacing(results_mm, spacing_file)
    
    # Save all crack data
    specimen_instance.save_cracks(cracks=cracks_90, folder_name=test_dir, file_name="cracks_90.pkl")
    specimen_instance.save_cracks(cracks=cracks_0, folder_name=test_dir, file_name="cracks_0.pkl")
    specimen_instance.save_cracks(cracks=filtered_cracks, folder_name=test_dir, file_name="filtered_cracks_90.pkl")

    # Assertions
    assert len(cracks_90) > 0, "No 90° cracks detected"
    assert len(cracks_0) > 0, "No 0° cracks detected"
    assert os.path.exists(spacing_file), "Spacing results file should be created"
 
def test_visualization_and_plotting(specimen_instance):
    """Test visualization capabilities"""
    test_dir = TEST_DIRS['visualization']
    
    # Load previously saved cracks
    cracks_90 = specimen_instance.load_cracks(os.path.join(TEST_DIRS['crossply_analysis'], "cracks_90.pkl"))
    cracks_0 = specimen_instance.load_cracks(os.path.join(TEST_DIRS['crossply_analysis'], "cracks_0.pkl"))

    # Create combined visualization
    joined_cracks = specimen_instance.join_cracks(cracks_90, cracks_0)
    
    # Plot last frame with all cracks
    fig, ax = specimen_instance.plot_cracks(
        image=specimen_instance.image_stack_cut[-1],
        cracks=joined_cracks[-1],
        color='red'
    )
    fig.savefig(os.path.join(test_dir, "all_cracks_combined.png"))
    plt.close(fig)

    # Create comparative plot with different colors
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot transverse cracks in red
    _, ax = specimen_instance.plot_cracks(
        image=specimen_instance.image_stack_cut[-1],
        cracks=cracks_90[-1],
        color='red',
        background_flag=False,
        linewidth=1.0,
        ax=ax
    )

    # Plot longitudinal cracks in blue
    _, ax = specimen_instance.plot_cracks(
        image=specimen_instance.image_stack_cut[-1],
        cracks=cracks_0[-1],
        color='blue',
        background_flag=False,
        linewidth=1.0,
        ax=ax
    )

    ax.set_xlabel('x [px]', fontsize=12)
    ax.set_ylabel('y [px]', fontsize=12)
    ax.set_title('Transverse (red) vs Longitudinal (blue) Cracks', fontsize=14)
    ax.grid(False)
    
    fig.savefig(os.path.join(test_dir, "crack_orientation_comparison.png"))
    plt.close(fig)

    assert os.path.exists(os.path.join(test_dir, "all_cracks_combined.png"))
    assert os.path.exists(os.path.join(test_dir, "crack_orientation_comparison.png"))

def test_postprocessing_and_filtering(specimen_instance):
    """Test advanced post-processing and filtering"""
    test_dir = TEST_DIRS['postprocessing']
    
    # Load cracks
    cracks_90 = specimen_instance.load_cracks(os.path.join(TEST_DIRS['crossply_analysis'], "cracks_90.pkl"))
    
    # Test different filtering parameters
    w = specimens['sample-1']['dimensions']['width'] * specimens['sample-1']['scale']
    
    # Test with different thresholds
    thresholds = [0.15*w, 0.20*w, 0.30*w]
    
    for i, threshold in enumerate(thresholds):
        processed_data, filtered_cracks = specimen_instance.crack_filtering_postprocessing(
            cracks_90, 50, threshold, remove_outliers=True
        )
        
        # Save filtered results
        specimen_instance.save_cracks(
            cracks=filtered_cracks, 
            folder_name=test_dir, 
            file_name=f"filtered_cracks_threshold_{i+1}.pkl"
        )
        
        # Convert and export spacing results
        results_mm = specimen_instance.pixels_to_length(processed_data)
        specimen_instance.export_crack_spacing(
            results_mm, 
            os.path.join(test_dir, f"spacing_results_threshold_{i+1}.csv")
        )
        
        # Create visualization of filtered cracks
        if len(filtered_cracks) > 0:
            fig, ax = specimen_instance.plot_cracks(
                image=specimen_instance.image_stack_cut[-1],
                cracks=filtered_cracks[-1],
                color='green',
                background_flag=True
            )
            fig.savefig(os.path.join(test_dir, f"filtered_cracks_visualization_{i+1}.png"))
            plt.close(fig)

    # Verify files were created
    assert len(os.listdir(test_dir)) > 0, "Post-processing should create output files"

def test_comprehensive_report():
    """Creates a test report"""
    report_dir = os.path.join(TESTS_BASE_DIR, "00_test_report")
    os.makedirs(report_dir, exist_ok=True)
    
    # Create a simple summary report
    with open(os.path.join(report_dir, "test_summary.md"), "w") as f:
        f.write("# DelaDect Test Summary\n\n")
        f.write(f"**Test Run:** {TEST_TIMESTAMP}\n\n")
        f.write("## Test Structure\n\n")
        f.write("| Test Type | Description | Output Location |\n")
        f.write("|-----------|-------------|-----------------|\n")
        for test_name, test_path in TEST_DIRS.items():
            f.write(f"| {test_name} | Comprehensive testing | `{os.path.basename(test_path)}` |\n")
    
    print(f"\nAll tests completed successfully!")
    print(f"Results saved in: {TESTS_BASE_DIR}")
    print(f"Test report available in: {report_dir}")

# Run the report function after all tests
@pytest.fixture(scope="session", autouse=True)
def final_report(request):
    """Generate final report after all tests"""
    request.addfinalizer(test_comprehensive_report)