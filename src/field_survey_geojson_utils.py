import geopandas as gpd
import logging

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load field survey data
def load_field_survey_geojson(path):
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
def clean_field_survey_geojson(field_survey, drop_columns=None):
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
def report_field_survey_geojson_missing_values(df, save_path=None):
    """
    Reports missing values in the data and optionally saves the report to a file.

    Parameters:
    - df (DataFrame): The data to analyze.
    - save_path (str, optional): Path to save the missing values report as a CSV file. Default is None.

    Returns:
    - DataFrame: Summary of missing values.
    """
    logger.info("Checking for missing values.")
    missing_values_table = df.isnull().sum().reset_index()
    missing_values_table.columns = ['Column', 'Missing count']
    missing_values_table['Missing percentage'] = (missing_values_table['Missing count'] / len(df)) * 100

    # Log missing values summary
    logger.info("Missing values summary:")
    logger.info(f"\n{missing_values_table}")

    # Save the missing values report if a path is provided
    if save_path:
        missing_values_table.to_csv(save_path, index=False)
        logger.info(f"Missing values report saved to {save_path}")

    return missing_values_table

