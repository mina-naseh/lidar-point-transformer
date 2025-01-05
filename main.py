import os
import logging
from src.field_survey_geojson_utils import (
    load_field_survey_geojson,
    clean_field_survey_geojson,
    report_field_survey_geojson_missing_values
)
from src.field_survey_visualization import (
    plot_geojson_species_map,
    plot_field_survey_subplots,
    plot_species_bar_chart,
    plot_field_density
)
from src.point_cloud_utils import (
    process_and_visualize_multiple_point_clouds
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
PLOTS_DIR = "./plots"
FIELD_SURVEY_PATH = os.path.join(DATA_DIR, "field_survey.geojson")
TIF_DIR = os.path.join(DATA_DIR, "ortho")
LAS_DIR = os.path.join(DATA_DIR, "als")
DROP_COLUMNS = ["tree_no"]  # Columns to drop from field survey

# Ensure Output Directories Exist
os.makedirs(PLOTS_DIR, exist_ok=True)


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
        field_survey = load_field_survey_geojson(FIELD_SURVEY_PATH)

        logger.info("Cleaning field survey data...")
        field_survey_cleaned = clean_field_survey_geojson(field_survey, DROP_COLUMNS)

        logger.info("Reporting missing values...")
        missing_values_log_file = os.path.join(LOGS_DIR, "missing_values.csv")
        report_field_survey_geojson_missing_values(
            field_survey_cleaned, save_path=missing_values_log_file
        )

        # --- Step 2: Visualize Field Survey ---
        logger.info("Plotting species count...")
        plot_species_bar_chart(
            field_survey_cleaned,
            species_col="species",
            save_path=os.path.join(PLOTS_DIR, "species_counts.png")
        )

        logger.info("Plotting density plots...")
        plot_field_density(
            field_survey_cleaned,
            columns=["d1", "d2", "dbh"],
            save_path=os.path.join(PLOTS_DIR, "density_plots.png")
        )

        logger.info("Plotting geographic map of species...")
        plot_geojson_species_map(
            field_survey_cleaned,
            species_col="species",
            save_path=os.path.join(PLOTS_DIR, "field_survey_map.png")
        )

        logger.info("Plotting individual rectangles for plots...")
        plot_field_survey_subplots(
            field_survey_cleaned,
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

        logger.info("Workflow complete!")

    except Exception as e:
        logger.error(f"An error occurred during the workflow: {e}", exc_info=True)


if __name__ == "__main__":
    main()
