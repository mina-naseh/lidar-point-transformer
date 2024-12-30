import geopandas as gpd
import logging

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load field survey data
def load_field_survey(path):
    """
    Loads the field survey data from the specified path.

    Parameters:
    - path (str): Path to the GeoJSON file.

    Returns:
    - GeoDataFrame: The loaded field survey data.
    """
    logger.info(f"Loading field survey data from {path}")
    field_survey = gpd.read_file(path)
    logger.info(f"Loaded {len(field_survey)} rows of data.")
    return field_survey

# Drop unnecessary columns
def clean_field_survey(field_survey, drop_columns=None):
    """
    Cleans the field survey data by dropping unnecessary columns.

    Parameters:
    - field_survey (GeoDataFrame): The field survey data.
    - drop_columns (list, optional): List of column names to drop.

    Returns:
    - GeoDataFrame: The cleaned data.
    """
    if drop_columns:
        logger.info(f"Dropping columns: {drop_columns}")
        field_survey = field_survey.drop(columns=drop_columns)
    return field_survey

# Check and report missing values
def report_missing_values(df):
    """
    Reports missing values in the data.

    Parameters:
    - df (DataFrame): The data to analyze.

    Returns:
    - None: Prints a summary of missing values.
    """
    logger.info("Checking for missing values.")
    missing_values_table = df.isnull().sum().reset_index()
    missing_values_table.columns = ['Column', 'Missing count']
    missing_values_table['Missing percentage'] = (missing_values_table['Missing count'] / len(df)) * 100

    # Log missing values summary
    logger.info("Missing values summary:")
    logger.info(f"\n{missing_values_table}")

    return missing_values_table
