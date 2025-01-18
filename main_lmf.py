import os
import shutil
import logging
import geopandas as gpd
from src.lmf_utils import (
    process_all_las_files_with_ground_truth
)

DATA_DIR = "./data"
ALS_PREPROCESSED_DIR = os.path.join(DATA_DIR, "als_preprocessed")
RESULTS_DIR = "./results_lmf"
FIELD_SURVEY_PATH = os.path.join(DATA_DIR, "field_survey.geojson")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def setup_results_directory(directory):
    """
    Ensures that the results directory exists and is empty.

    Args:
        directory (str): Path to the results directory.
    """
    if os.path.exists(directory):
        logger.info(f"Clearing existing directory: {directory}")
        shutil.rmtree(directory)
    os.makedirs(directory)
    logger.info(f"Created directory: {directory}")

def main():
    setup_results_directory(RESULTS_DIR)

    if not os.path.exists(FIELD_SURVEY_PATH):
        logger.error(f"Ground truth file not found at {FIELD_SURVEY_PATH}. Exiting...")
        return
    ground_truth_data = gpd.read_file(FIELD_SURVEY_PATH)

    logger.info("Starting Local Maxima Filtering pipeline...")
    metrics_summary = process_all_las_files_with_ground_truth(
        las_dir=ALS_PREPROCESSED_DIR,
        ground_truth_data=ground_truth_data,
        save_dir=RESULTS_DIR,
        max_distance=5.0,
        max_height_difference=3.0, 
        window_size=2.0  
    )

    if metrics_summary.empty:
        logger.warning("No metrics were generated. Please check the input data.")
    else:
        logger.info("Detection metrics summary saved to results directory.")

if __name__ == "__main__":
    main()
