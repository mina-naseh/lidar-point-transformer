import os
import shutil
import logging
from src.pre_visualization import process_data

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

DATA_DIR = "./data"
OUTPUT_DIR = "./results_pre"

def setup_output_directory(directory):
    """
    Ensures that the output directory exists and is empty.

    Args:
        directory (str): Path to the output directory.
    """
    if os.path.exists(directory):
        logger.info(f"Clearing existing directory: {directory}")
        shutil.rmtree(directory)
    os.makedirs(directory)
    logger.info(f"Created directory: {directory}")

def main():
    """
    Main function to process data and generate visualizations.
    """
    logger.info("Setting up output directory...")
    setup_output_directory(OUTPUT_DIR)

    logger.info("Starting data processing workflow...")
    process_data(data_dir=DATA_DIR, output_dir=OUTPUT_DIR)

    logger.info("Workflow complete!")

if __name__ == "__main__":
    main()
