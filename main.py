import os
import logging
from src.preprocessing import (
    load_field_survey,
    clean_field_survey,
    report_missing_values
)
from src.visualization import (
    plot_field_survey_map,
    plot_individual_rectangles,
    plot_species_counts,
    plot_density
)
from src.point_cloud_utils import (
    visualize_raster_images,
    visualize_point_cloud
)

# Configure logging
LOGS_DIR = "./logs"
os.makedirs(LOGS_DIR, exist_ok=True)  # Ensure the logs directory exists
LOG_FILE = os.path.join(LOGS_DIR, "workflow.log")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler(LOG_FILE, mode="w")  # Log to file
    ]
)
logger = logging.getLogger(__name__)

# Define constants
DATA_DIR = "./data"
PLOTS_DIR = "./plots"
FIELD_SURVEY_PATH = os.path.join(DATA_DIR, "field_survey.geojson")
TIF_DIR = os.path.join(DATA_DIR, "ortho")
LAS_DIR = os.path.join(DATA_DIR, "als")
DROP_COLUMNS = ["tree_no"]  # Columns to drop from field survey

# Ensure output directories exist
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


def main():
    try:
        logger.info("Starting the main workflow...")

        # Check if input files and directories exist
        if not os.path.exists(FIELD_SURVEY_PATH):
            logger.error(f"Field survey file not found at {FIELD_SURVEY_PATH}")
            return
        if not os.path.exists(TIF_DIR):
            logger.error(f"TIF directory not found at {TIF_DIR}")
            return
        if not os.path.exists(LAS_DIR):
            logger.error(f"LAS directory not found at {LAS_DIR}")
            return

        # Step 1: Load and preprocess field survey data
        logger.info("Loading field survey data...")
        field_survey = load_field_survey(FIELD_SURVEY_PATH)

        logger.info("Cleaning field survey data...")
        field_survey_cleaned = clean_field_survey(field_survey, DROP_COLUMNS)

        logger.info("Reporting missing values...")
        missing_values = report_missing_values(field_survey_cleaned)

        # Save missing values report (optional)
        missing_values_file = os.path.join(PLOTS_DIR, "missing_values.csv")
        missing_values.to_csv(missing_values_file, index=False)
        logger.info(f"Missing values report saved to {missing_values_file}")

        # Step 2: Plot field survey
        logger.info("Plotting species count...")
        plot_species_counts(
            field_survey_cleaned,
            species_col="species",
            save_path=os.path.join(PLOTS_DIR, "species_counts.png")
        )

        logger.info("Plotting density plots...")
        plot_density(
            field_survey_cleaned,
            columns=["d1", "d2", "dbh"],
            save_path=os.path.join(PLOTS_DIR, "density_plots.png")
        )

        logger.info("Plotting field survey map...")
        plot_field_survey_map(
            field_survey_cleaned,
            species_col="species",
            save_path=os.path.join(PLOTS_DIR, "field_survey_map.png")
        )

        logger.info("Plotting individual rectangles...")
        plot_individual_rectangles(
            field_survey_cleaned,
            plot_col="plot",
            save_path=os.path.join(PLOTS_DIR, "individual_rectangles.png")
        )

        # Step 3: Visualize raster images and point clouds
        logger.info("Visualizing raster images...")
        visualize_raster_images(
            TIF_DIR,
            save_path=os.path.join(PLOTS_DIR, "raster_images.png")
        )

        logger.info("Visualizing point cloud data...")
        visualize_point_cloud(
            LAS_DIR,
            height_threshold=15,
            save_path=os.path.join(PLOTS_DIR, "point_cloud_visualization.png")
        )

        logger.info("Workflow complete!")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()
