import matplotlib.pyplot as plt
import logging
import seaborn as sns
import math
import os
import rasterio

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def save_plot(save_path):
    """
    Saves the current plot to a file.

    Parameters:
    - save_path (str): Path to save the plot.
    """
    try:
        plt.savefig(save_path, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save plot to {save_path}: {e}")
    finally:
        plt.close()

def plot_geojson_species_map(gdf, species_col='species', save_path=None):
    """
    Plots a geographic map of trees colored by species and saves the plot.
    """
    if gdf.empty:
        logger.warning("GeoDataFrame is empty. Skipping geographic map plotting.")
        return

    try:
        gdf = gdf.to_crs(epsg=4326)  # Convert to WGS84
        ax = gdf.plot(column=species_col, cmap='viridis', legend=True, figsize=(15, 15), markersize=1)
        plt.title('Geographic Plot of Trees Colored by Species', fontsize=21)

        legend = ax.get_legend()
        if legend:
            for label in legend.get_texts():
                label.set_fontsize(16)

        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.tight_layout()
        save_plot(save_path)
    except Exception as e:
        logger.error(f"Error during species map plotting: {e}")

def plot_field_survey_subplots(gdf, plot_col='plot', save_path=None):
    """
    Creates a subplot for each unique plot ID, showing geographic tree distribution, and saves the plot.
    """
    if gdf.empty:
        logger.warning("GeoDataFrame is empty. Skipping subplot creation.")
        return

    try:
        unique_plots = sorted(gdf[plot_col].unique())
        logger.info(f"Creating subplots for {len(unique_plots)} unique plots.")

        num_cols = min(5, len(unique_plots))
        num_rows = math.ceil(len(unique_plots) / num_cols)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 5), squeeze=False)

        for i, plot_id in enumerate(unique_plots):
            ax = axes[i // num_cols, i % num_cols]
            subset = gdf[gdf[plot_col] == plot_id]
            subset.plot(column='species', cmap='viridis', legend=False, ax=ax, markersize=9)
            ax.set_title(f"Plot {plot_id}", fontsize=14)
            ax.set_xlim(subset.total_bounds[[0, 2]])
            ax.set_ylim(subset.total_bounds[[1, 3]])
            ax.axis('off')

        for ax in axes.flatten()[len(unique_plots):]:
            ax.set_visible(False)

        plt.tight_layout()
        save_plot(save_path)
    except Exception as e:
        logger.error(f"Error during subplot plotting: {e}")

def plot_species_bar_chart(df, species_col='species', save_path=None):
    """
    Creates a bar plot for species counts and saves the plot.
    """
    if df.empty:
        logger.warning("The DataFrame is empty. Skipping species count plot.")
        return

    try:
        species_counts = df[species_col].value_counts().reset_index()
        species_counts.columns = [species_col, 'Count']
        logger.info(f"Plotting bar chart for {len(species_counts)} species.")

        plt.figure(figsize=(15, 6))
        sns.barplot(
            x='Count',
            y=species_col,
            data=species_counts,
            palette="viridis"
        )
        plt.title("Species Count", fontsize=16)
        plt.xlabel("Count", fontsize=12)
        plt.ylabel("Species", fontsize=12)
        plt.tight_layout()
        save_plot(save_path)
    except Exception as e:
        logger.error(f"Error during species bar chart plotting: {e}")

def plot_field_density(df, columns, save_path=None):
    """
    Creates density plots for specified columns and saves the plot.
    """
    if df.empty:
        logger.warning("The DataFrame is empty. Skipping density plot.")
        return

    try:
        logger.info(f"Creating density plots for columns: {columns}")
        num_cols = len(columns)
        plt.figure(figsize=(num_cols * 6, 5))
        for i, col in enumerate(columns, 1):
            plt.subplot(1, num_cols, i)
            sns.kdeplot(df[col], fill=True)
            plt.title(f"Density Plot for {col}", fontsize=14)
            plt.xlabel(col, fontsize=12)
            plt.ylabel("Density", fontsize=12)

        plt.tight_layout()
        save_plot(save_path)
    except Exception as e:
        logger.error(f"Error during density plot creation: {e}")

def visualize_raster_images(tif_dir, num_plots=10, save_path=None):
    """
    Visualizes raster images from a specified directory and saves the plot.
    """
    if not os.path.exists(tif_dir):
        logger.error(f"Directory does not exist: {tif_dir}")
        return

    tiff_files = [os.path.join(tif_dir, f'plot_{i:02d}.tif') for i in range(1, num_plots + 1)]
    tiff_files = [f for f in tiff_files if os.path.exists(f)]

    if not tiff_files:
        logger.warning(f"No valid raster files found in directory: {tif_dir}")
        return

    logger.info(f"Visualizing {len(tiff_files)} raster images.")
    num_cols = min(5, len(tiff_files))
    num_rows = math.ceil(len(tiff_files) / num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 5), squeeze=False)

    for i, tiff_file in enumerate(tiff_files):
        ax = axes[i // num_cols, i % num_cols]
        try:
            with rasterio.open(tiff_file) as src:
                img = src.read(1)
                ax.imshow(img, cmap='gist_earth')
                ax.set_title(os.path.basename(tiff_file).split('.')[0], fontsize=12)
                ax.axis('off')
        except Exception as e:
            logger.warning(f"Error reading raster file {tiff_file}: {e}")
            ax.set_visible(False)

    for ax in axes.flatten()[len(tiff_files):]:
        ax.set_visible(False)

    plt.tight_layout()
    save_plot(save_path)


import numpy as np
import laspy
import logging
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import os
import glob

import scipy.spatial
import geopandas as gpd
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist
import pandas as pd


# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)




def load_las_file(file_path):
    """
    Loads a LAS file and returns its point cloud data.

    Parameters:
    - file_path (str): Path to the LAS file.

    Returns:
    - numpy.ndarray: Point cloud data (X, Y, Z, classification).
    """
    try:
        las = laspy.read(file_path)

        # Extract X, Y, Z, and classification
        points = np.vstack((las.x, las.y, las.z, las.classification)).T

        logger.info(f"Loaded {points.shape[0]} points from {file_path}")
        return points
    except Exception as e:
        logger.error(f"Error loading LAS file {file_path}: {e}")
        return np.array([])  # Return an empty array if an error occurs



def filter_points_by_height(points, height_threshold=None, percentile=None):
    """
    Filters points based on a height threshold or a percentile of Z-values.

    Parameters:
    - points (numpy.ndarray): Point cloud data (X, Y, Z, classification).
    - height_threshold (float, optional): Minimum Z value to retain points.
    - percentile (float, optional): Percentile of Z-values to compute threshold. If provided, overrides height_threshold.

    Returns:
    - numpy.ndarray: Filtered point cloud data.
    """
    if points.shape[1] < 3:
        logger.error("Points array does not contain enough columns for filtering.")
        return points

    if percentile is not None:
        z_threshold = np.percentile(points[:, 2], percentile)  # Percentile of Z-values
        logger.info(f"Using percentile-based height threshold: {z_threshold:.2f}")
    elif height_threshold is not None:
        z_threshold = height_threshold  # Fixed height threshold
        logger.info(f"Using fixed height threshold: {z_threshold:.2f}")
    else:
        logger.warning("No height threshold or percentile provided. Returning all points.")
        return points

    # Filter points based on Z (height)
    filtered_points = points[points[:, 2] >= z_threshold]
    logger.info(f"Filtered points: {filtered_points.shape[0]} retained out of {points.shape[0]}")

    return filtered_points



def remove_noise_with_dbscan(points, eps=1.0, min_samples=5):
    """
    Removes noise from point clouds using DBSCAN clustering.

    Parameters:
    - points (numpy.ndarray): Point cloud data (X, Y, Z, classification).
    - eps (float): The maximum distance between two samples for them to be in the same cluster.
    - min_samples (int): The minimum number of points required to form a dense region.

    Returns:
    - numpy.ndarray: Noise-free point cloud data.
    """
    if points.size == 0:
        logger.warning("No points provided for DBSCAN. Returning empty array.")
        return points

    try:
        # Apply DBSCAN on spatial coordinates only (X, Y, Z)
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points[:, :3])
    except ValueError as e:
        logger.error(f"DBSCAN clustering failed: {e}")
        return points

    # Retain only points with valid cluster labels (label >= 0)
    labels = clustering.labels_
    filtered_points = points[labels >= 0]

    logger.info(f"DBSCAN retained {filtered_points.shape[0]} points out of {points.shape[0]}")
    return filtered_points


# --- Visualization Functions ---

def visualize_point_cloud(points, title="Point Cloud", color_by="height", save_path=None):
    """
    Visualizes a 3D point cloud.

    Parameters:
    - points (numpy.ndarray): Point cloud data (X, Y, Z, classification).
    - title (str): Title for the plot. Default is 'Point Cloud'.
    - color_by (str): Attribute to color the points by. Options: 'height' or 'classification'. Default is 'height'.
    - save_path (str, optional): Path to save the plot. Default is None.

    Returns:
    - None: Displays or saves the plot.
    """
    if points.size == 0:
        logger.warning("No points to visualize.")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Determine color mapping based on the chosen attribute
    if color_by == "classification" and points.shape[1] > 3:
        color_data = points[:, 3]  # Classification
        cmap = "tab10"  # Discrete colormap for classification
        logger.info("Coloring points by classification.")
    else:
        color_data = points[:, 2]  # Height (Z)
        cmap = "viridis"  # Continuous colormap for height
        logger.info("Coloring points by height.")

    # Create 3D scatter plot
    scatter = ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=color_data, s=1, cmap=cmap
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title(title)

    # Add colorbar for the scatter plot
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label(color_by.capitalize())

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        logger.info(f"{title} plot saved to {save_path}")
    else:
        plt.show()

    plt.close()



def visualize_filtered_point_cloud(file_path, height_threshold=None, apply_dbscan=False, eps=1.0, min_samples=5, color_by="height", save_path=None):
    """
    Visualizes a filtered 3D point cloud from a LAS file.

    Parameters:
    - file_path (str): Path to the LAS file.
    - height_threshold (float, optional): Minimum height for filtering. Default is None.
    - apply_dbscan (bool): Whether to apply DBSCAN clustering. Default is False.
    - eps (float): Maximum distance for DBSCAN clustering. Default is 1.0.
    - min_samples (int): Minimum samples for DBSCAN clustering. Default is 5.
    - color_by (str): Attribute to color points by ('height' or 'classification'). Default is 'height'.
    - save_path (str, optional): Path to save the plot. Default is None.

    Returns:
    - None: Displays or saves the plot.
    """
    points = load_las_file(file_path)
    if points.size == 0:
        logger.warning("No points to visualize.")
        return

    # Filter points by height threshold
    if height_threshold is not None:
        original_count = len(points)
        points = filter_points_by_height(points, height_threshold)
        logger.info(f"Applied height threshold {height_threshold}. Retained {len(points)} out of {original_count} points.")
        if points.size == 0:
            logger.warning("No points remaining after height filtering.")
            return

    # Apply DBSCAN noise removal
    if apply_dbscan:
        original_count = len(points)
        points = remove_noise_with_dbscan(points, eps, min_samples)
        logger.info(f"Applied DBSCAN (eps={eps}, min_samples={min_samples}). Retained {len(points)} out of {original_count} points.")
        if points.size == 0:
            logger.warning("No points remaining after DBSCAN filtering.")
            return

    # Visualize the filtered point cloud
    visualize_point_cloud(
        points,
        title=f"Filtered Point Cloud - {os.path.basename(file_path)}",
        color_by=color_by,
        save_path=save_path
    )



def process_and_visualize_multiple_point_clouds(
    las_dir, save_dir=None, apply_dbscan=False, eps=1.0, min_samples=5, percentile=None, color_by="height"
):
    """
    Processes and visualizes multiple LAS files with optional filtering and DBSCAN.

    Parameters:
    - las_dir (str): Directory containing LAS files.
    - save_dir (str, optional): Directory to save plots.
    - apply_dbscan (bool): Whether to apply DBSCAN clustering.
    - eps (float): Maximum distance for DBSCAN clustering.
    - min_samples (int): Minimum samples for DBSCAN clustering.
    - percentile (float, optional): Percentile of Z-values to compute threshold.
    - color_by (str): Attribute to color points by ('height' or 'classification'). Default is 'height'.

    Returns:
    - None: Displays or saves the plots.
    """

    # Collect all LAS files from the directory
    las_files = [os.path.join(las_dir, f) for f in os.listdir(las_dir) if f.endswith(".las")]
    if not las_files:
        logger.warning(f"No LAS files found in directory: {las_dir}")
        return

    # Process each LAS file
    for file_path in sorted(las_files):
        # Extract plot number from the file name (e.g., plot_01 from plot_01.las)
        try:
            plot_number = int(os.path.basename(file_path).split("_")[1].split(".")[0])  # Extract '01', '02', etc.
        except ValueError:
            logger.error(f"Could not extract plot number from file: {file_path}. Skipping...")
            continue

        logger.info(f"Processing plot_{plot_number} from file {file_path}...")

        points = load_las_file(file_path)
        if points.size == 0:
            logger.warning(f"File {file_path} is empty. Skipping...")
            continue

        # Apply percentile-based Z-value threshold if specified
        if percentile is not None:
            original_count = len(points)
            z_threshold = np.percentile(points[:, 2], percentile)
            logger.info(f"Using {percentile}th percentile as Z-value threshold: {z_threshold:.3f}")
            points = filter_points_by_height(points, z_threshold)
            logger.info(f"Retained {len(points)} points out of {original_count} after height filtering.")
            if points.size == 0:
                logger.warning(f"No points remaining after height filtering for plot_{plot_number}. Skipping...")
                continue

        # Apply DBSCAN clustering if enabled
        if apply_dbscan:
            original_count = len(points)
            points = remove_noise_with_dbscan(points, eps=eps, min_samples=min_samples)
            logger.info(f"Retained {len(points)} points out of {original_count} after DBSCAN filtering.")
            if points.size == 0:
                logger.warning(f"No points remaining after DBSCAN filtering for plot_{plot_number}. Skipping...")
                continue

        # Visualize and optionally save the filtered point cloud
        plot_title = f"Filtered Point Cloud - plot_{plot_number}"
        save_path = os.path.join(save_dir, f"plot_{plot_number}.png") if save_dir else None
        visualize_point_cloud(points, title=plot_title, color_by=color_by, save_path=save_path)




# --- LMF Functions ---

def normalize_cloud_height(points, ground_points):
    """
    Normalizes the Z-values of the point cloud relative to the ground level.

    Parameters:
    - points (numpy.ndarray): Full point cloud data (X, Y, Z, classification).
    - ground_points (numpy.ndarray): Ground points data (X, Y, Z).

    Returns:
    - numpy.ndarray: Point cloud with normalized heights (X, Y, Z, classification).
    """
    if ground_points.size == 0:
        logger.warning("No ground points provided. Returning original points.")
        return points

    try:
        logger.info(f"Normalizing heights using {len(ground_points)} ground points...")

        # Interpolate ground level based on ground points
        ground_level = griddata(
            points=ground_points[:, :2],  # Use X, Y of ground points
            values=ground_points[:, 2],  # Use Z (height) of ground points
            xi=points[:, :2],            # Interpolate for all points' X, Y
            method="nearest"
        )

        # Normalize the Z-values of points
        normalized_points = points.copy()
        normalized_points[:, 2] -= ground_level  # Subtract ground level from Z

        logger.info("Height normalization complete.")
        return normalized_points

    except Exception as e:
        logger.error(f"Error during height normalization: {e}")
        return points  # Return original points in case of failure



import geopandas as gpd
import logging
import numpy as np

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
    try:
        logger.info(f"Loading field survey data from {path}")
        field_survey = gpd.read_file(path)
        logger.info(f"Loaded {len(field_survey)} rows of data.")
        return field_survey
    except Exception as e:
        logger.error(f"Failed to load GeoJSON file: {e}")
        return gpd.GeoDataFrame()  # Return an empty GeoDataFrame for robustness


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
        try:
            field_survey.drop(columns=drop_columns, inplace=True)
        except KeyError as e:
            logger.warning(f"Some columns to drop were not found: {e}")
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
        try:
            missing_values_table.to_csv(save_path, index=False)
            logger.info(f"Missing values report saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save missing values report: {e}")

    return missing_values_table


# Extract ground truth data for a specific plot
def get_plot_ground_truth(field_survey, plot_id):
    """
    Extracts ground truth data for a specific plot.

    Parameters:
    - field_survey (GeoDataFrame): The field survey data.
    - plot_id (int): The ID of the plot.

    Returns:
    - np.ndarray: Ground truth tree coordinates and heights for the plot.
    """
    logger.info(f"Extracting ground truth data for plot {plot_id}.")
    if plot_id not in field_survey["plot"].unique():
        logger.warning(f"Plot ID {plot_id} not found in the dataset.")
        return np.array([])

    plot_data = field_survey[field_survey["plot"] == plot_id]

    # Ensure only trees with valid coordinates and heights are included
    ground_truth = plot_data[["geometry", "height"]].dropna()
    ground_truth_array = ground_truth.apply(
        lambda row: [row.geometry.x, row.geometry.y, row.height], axis=1
    ).to_list()

    logger.info(f"Found {len(ground_truth_array)} ground truth trees for plot {plot_id}.")
    return np.array(ground_truth_array)


# Load, clean, and prepare the field survey data
def process_field_survey_geojson(
    path, drop_columns=None, missing_values_report_path=None
):
    """
    Loads, cleans, and prepares the field survey data.

    Parameters:
    - path (str): Path to the GeoJSON file.
    - drop_columns (list, optional): Columns to drop from the data.
    - missing_values_report_path (str, optional): Path to save missing values report.

    Returns:
    - GeoDataFrame: Cleaned and processed field survey data.
    """
    field_survey = load_field_survey_geojson(path)
    if field_survey.empty:
        logger.error("Field survey data is empty after loading. Exiting process.")
        return field_survey

    # Report missing values
    if missing_values_report_path:
        report_field_survey_geojson_missing_values(
            field_survey, save_path=missing_values_report_path
        )

    # Clean field survey data
    if drop_columns:
        field_survey = clean_field_survey_geojson(field_survey, drop_columns)

    return field_survey
