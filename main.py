import os
import logging
import geopandas as gpd
from src.field_survey_geojson_utils import (
    load_field_survey_geojson,
    clean_field_survey_geojson,
    report_field_survey_geojson_missing_values,
    process_field_survey_geojson,
    get_plot_ground_truth,
)
from src.field_survey_visualization import (
    plot_geojson_species_map,
    plot_field_survey_subplots,
    plot_species_bar_chart,
    plot_field_density
)
from src.point_cloud_utils import (
    process_and_visualize_multiple_point_clouds,
    normalize_cloud_height, 
    local_maxima_filter, 
    process_point_cloud_with_lmf,
    process_and_visualize_multiple_point_clouds_with_lmf,
    process_all_las_files_with_ground_truth
)
from src.orthophotos_utils import (
    visualize_raster_images
)

# --- Configure Logging ---
LOGS_DIR = "./logs"
LOG_FILE = os.path.join(LOGS_DIR, "workflow.log")

os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode="w")
    ]
)
logger = logging.getLogger(__name__)

# --- Constants ---
DATA_DIR = "./data"
PLOTS_DIR = "./visualizations"
RESULTS_DIR = "./results"
FIELD_SURVEY_PATH = os.path.join(DATA_DIR, "field_survey.geojson")
TIF_DIR = os.path.join(DATA_DIR, "ortho")
LAS_DIR = os.path.join(DATA_DIR, "als")
DROP_COLUMNS = ["tree_no"]  # Columns to drop from field survey

# Ensure Output Directories Exist
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# --- Utility Functions ---
def validate_path(path, path_type="file"):
    """
    Validates if a file or directory exists.

    Parameters:
    - path (str): Path to validate.
    - path_type (str): Either "file" or "directory". Default is "file".

    Returns:
    - bool: True if the path exists, False otherwise.
    """
    if path_type == "file" and not os.path.isfile(path):
        logger.error(f"Required file not found: {path}")
        return False
    elif path_type == "directory" and not os.path.isdir(path):
        logger.error(f"Required directory not found: {path}")
        return False
    return True


def main():
    try:
        logger.info("Starting the main workflow...")

        # Validate paths
        if not all([
            validate_path(FIELD_SURVEY_PATH, path_type="file"),
            validate_path(TIF_DIR, path_type="directory"),
            validate_path(LAS_DIR, path_type="directory")
        ]):
            logger.error("One or more required paths are missing. Exiting workflow.")
            return

        # --- Step 1: Load and Preprocess Field Survey Data ---
        logger.info("Loading field survey data...")
        field_survey = process_field_survey_geojson(
            path=FIELD_SURVEY_PATH,
            drop_columns=DROP_COLUMNS,
            missing_values_report_path=os.path.join(LOGS_DIR, "missing_values.csv"),
        )

        # --- Step 2: Visualize Field Survey ---
        logger.info("Plotting species count...")
        plot_species_bar_chart(
            field_survey,
            species_col="species",
            save_path=os.path.join(PLOTS_DIR, "species_counts.png")
        )

        logger.info("Plotting density plots...")
        plot_field_density(
            field_survey,
            columns=["d1", "d2", "dbh"],
            save_path=os.path.join(PLOTS_DIR, "density_plots.png")
        )

        logger.info("Plotting geographic map of species...")
        plot_geojson_species_map(
            field_survey,
            species_col="species",
            save_path=os.path.join(PLOTS_DIR, "field_survey_map.png")
        )

        logger.info("Plotting individual rectangles for plots...")
        plot_field_survey_subplots(
            field_survey,
            plot_col="plot",
            save_path=os.path.join(PLOTS_DIR, "individual_rectangles.png")
        )

        # --- Step 3: Visualize Raster Images ---
        logger.info("Visualizing raster images...")
        visualize_raster_images(
            TIF_DIR,
            save_path=os.path.join(PLOTS_DIR, "raster_images.png")
        )

        # --- Step 4: Process and Visualize Point Cloud Data ---
        logger.info("Processing and visualizing point cloud data...")
        process_and_visualize_multiple_point_clouds(
            las_dir=LAS_DIR,
            save_dir=PLOTS_DIR,
            apply_dbscan=True,
            eps=1.0,
            min_samples=5,
            percentile=5  # Use the 5th percentile as the threshold
        )

        # --- Step 5: Process Point Cloud Data with Local Maxima Filtering ---
        logger.info("Processing and visualizing point cloud data with Local Maxima Filtering...")
        process_and_visualize_multiple_point_clouds_with_lmf(
            las_dir=LAS_DIR,
            save_dir=RESULTS_DIR,
            apply_dbscan=True,  # Use DBSCAN for noise removal
            eps=1.0,            # DBSCAN epsilon value
            min_samples=5,      # DBSCAN minimum samples
            window_size=2.0,    # LMF window size
            height_threshold=3.0  # LMF height threshold
        )

        # --- Step 6: Match Detected Trees with Ground Truth and Calculate Metrics ---
        logger.info("Loading ground truth data...")
        ground_truth_data = gpd.read_file(FIELD_SURVEY_PATH)

        logger.info("Processing LAS files with ground truth matching and calculating metrics...")
        metrics_summary = process_all_las_files_with_ground_truth(
            las_dir=LAS_DIR,
            ground_truth_data=ground_truth_data,
            save_dir=RESULTS_DIR,
            max_distance=5.0,          # Maximum distance for matching
            max_height_difference=3.0, # Maximum height difference for matching
            window_size=2.0,           # Window size for LMF
            height_threshold=3.0       # Height threshold for LMF
        )

        # Save and log metrics
        if not metrics_summary.empty:
            metrics_summary_path = os.path.join(RESULTS_DIR, "detection_metrics_summary.csv")
            metrics_summary.to_csv(metrics_summary_path, index=False)
            logger.info(f"Metrics summary saved to {metrics_summary_path}.")
            logger.info(metrics_summary)
        else:
            logger.warning("No metrics were calculated.")

        logger.info("Workflow complete!")

    except Exception as e:
        logger.error(f"An error occurred during the workflow: {e}", exc_info=True)


if __name__ == "__main__":
    main()
