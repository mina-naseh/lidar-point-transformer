import os
import logging
from src.pre_visualization import process_data

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Define input and output directories
DATA_DIR = "./data"
OUTPUT_DIR = "./results_pre"

def main():
    """
    Main function to process data and generate visualizations.
    """
    logger.info("Starting data processing workflow...")
    process_data(data_dir=DATA_DIR, output_dir=OUTPUT_DIR)
    logger.info("Workflow complete!")

if __name__ == "__main__":
    main()
