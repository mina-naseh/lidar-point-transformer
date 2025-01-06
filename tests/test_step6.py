import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
)
from src.point_cloud_utils import process_all_las_files_with_ground_truth

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
LAS_DIR = os.path.join(DATA_DIR, "als")


def test_step6():
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


if __name__ == "__main__":
    # Run Step 6 only for testing
    test_step6()
