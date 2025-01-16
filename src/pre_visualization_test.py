import matplotlib.pyplot as plt
import logging
import seaborn as sns
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import laspy
import rasterio
from sklearn.preprocessing import MinMaxScaler

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# --- Utility Functions ---
def save_plot(save_path):
    """
    Saves the current plot to a file.

    Parameters:
    - save_path (str): Path to save the plot.
    """
    plt.savefig(save_path, bbox_inches="tight")
    logger.info(f"Plot saved to {save_path}")
    plt.close()

def inspect_geojson_data(gdf, save_path):
    """
    Inspects GeoJSON data for label distributions and missing values,
    saves the report, and visualizes species distribution.

    Parameters:
    - gdf (GeoDataFrame): The GeoDataFrame to inspect.
    - save_path (str): Path to save the summary reports and visualizations.
    """
    logger.info("Inspecting GeoJSON data...")

    # Create output directory if not exists
    os.makedirs(save_path, exist_ok=True)

    species_counts = gdf["species"].value_counts().reset_index()
    species_counts.columns = ["species", "count"]
    species_counts["Type"] = species_counts["species"].apply(
        lambda x: "Coniferous" if x in ["Fir", "Pine", "Spruce"] else "Deciduous"
    )
    species_counts.to_csv(os.path.join(save_path, "species_distribution.csv"), index=False)
    logger.info("Species distribution saved as species_distribution.csv")

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=species_counts,
        x="count",
        y="species",
        hue="Type",
        dodge=False,
        hue_order=["Coniferous", "Deciduous"],
        palette=["#2A5C03", "#DAA520"],
        saturation=1,
    )
    ax.set_title("Species Distribution in the Dataset", fontsize=16)
    ax.set_xlabel("Count", fontsize=12)
    ax.set_ylabel("Species", fontsize=12)
    ax.grid(axis="x", color="black", alpha=0.1)

    plot_path = os.path.join(save_path, "species_distribution.png")
    save_plot(plot_path)

def inspect_las_data(las_dir, save_path):
    """
    Inspects LAS files for basic statistics.

    Parameters:
    - las_dir (str): Directory containing LAS files.
    - save_path (str): Path to save the summary report.
    """
    logger.info("Inspecting LAS data...")
    las_files = [os.path.join(las_dir, f) for f in os.listdir(las_dir) if f.endswith(".las")]
    stats = []
    for file in las_files:
        las = laspy.read(file)
        points = np.vstack((las.x, las.y, las.z)).T
        stats.append({
            "file": file,
            "num_points": len(points),
            "min_height": points[:, 2].min(),
            "max_height": points[:, 2].max(),
        })
    stats_df = pd.DataFrame(stats)
    logger.info(f"LAS file statistics:\n{stats_df}")
    stats_df.to_csv(os.path.join(save_path, "las_stats.csv"), index=False)

# --- Visualization Functions ---
def plot_geojson_species_map(gdf, save_path):
    """
    Plots a geographic map of trees colored by species and saves the plot.
    """
    gdf = gdf.to_crs(epsg=4326)
    ax = gdf.plot(column="species", cmap="viridis", legend=True, figsize=(15, 15), markersize=1)
    plt.title("Tree Locations Colored by Species", fontsize=18)
    save_plot(save_path)

def visualize_raster_images(tif_dir, save_path):
    """
    Visualizes raster images in a 5x2 layout and saves the plot.

    Parameters:
    - tif_dir (str): Path to the directory containing raster files (.tif).
    - save_path (str): Path to save the combined plot.
    """
    logger.info(f"Visualizing raster images in {tif_dir}...")

    tiff_files = [os.path.join(tif_dir, f) for f in os.listdir(tif_dir) if f.endswith(".tif")]
    if not tiff_files:
        logger.warning(f"No raster files found in {tif_dir}.")
        return
    
    num_files = len(tiff_files)
    assert num_files <= 10, "The directory contains more than 10 raster files."

    num_cols = 5
    num_rows = 2

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5))

    axes = axes.flatten()

    for i, (tif_file, ax) in enumerate(zip(tiff_files, axes)):
        with rasterio.open(tif_file) as src:
            img = src.read()
            ax.imshow(np.rollaxis(img, 0, 3) / 255.0)
            ax.set_title(f"Raster: {os.path.basename(tif_file)}", fontsize=12)
            ax.axis("off")

    for ax in axes[len(tiff_files):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    logger.info(f"Raster images saved in 5x2 layout as {save_path}")
    plt.close()


# --- Normalization and Filtering ---
def normalize_geojson_heights(gdf):
    """
    Normalizes the height values in the GeoJSON file.

    Parameters:
    - gdf (GeoDataFrame): The GeoDataFrame to normalize.

    Returns:
    - GeoDataFrame: Normalized GeoDataFrame.
    """
    if "height" not in gdf.columns:
        logger.warning("No 'height' column in GeoJSON. Skipping normalization.")
        return gdf

    scaler = MinMaxScaler()
    gdf["normalized_height"] = scaler.fit_transform(gdf[["height"]])
    logger.info("Height normalization complete.")
    return gdf

def filter_las_by_height(las_file, min_height):
    """
    Filters points in a LAS file based on a minimum height threshold.

    Parameters:
    - las_file (str): Path to the LAS file.
    - min_height (float): Minimum height threshold.

    Returns:
    - numpy.ndarray: Filtered point cloud data.
    """
    las = laspy.read(las_file)
    points = np.vstack((las.x, las.y, las.z)).T
    filtered_points = points[points[:, 2] >= min_height]
    logger.info(f"Filtered {len(filtered_points)} points from {las_file}.")
    return filtered_points

def save_filtered_las(points, save_path):
    """
    Saves filtered LAS points to a new file.

    Parameters:
    - points (numpy.ndarray): Filtered point cloud data.
    - save_path (str): Path to save the filtered LAS file.
    """
    header = laspy.header.LasHeader(point_format=3, version="1.2")
    las = laspy.LasData(header)
    las.x, las.y, las.z = points[:, 0], points[:, 1], points[:, 2]
    las.write(save_path)
    logger.info(f"Filtered LAS file saved to {save_path}")



def plot_tree_type_distribution_by_plot(gdf, save_path):
    """
    Plots the distribution of coniferous and deciduous trees across plots and saves the plot.

    Parameters:
    - gdf (GeoDataFrame): The GeoDataFrame containing tree data.
    - save_path (str): Path to save the plot image.
    """
    logger.info("Visualizing tree type distribution by plot...")

    conifers = ["Fir", "Pine", "Spruce"] 
    gdf["Type"] = gdf["species"].map(lambda x: "Coniferous" if x in conifers else "Deciduous")

    plt.figure(figsize=(10, 6))
    ax = sns.histplot(
        data=gdf,
        y="plot",
        discrete=True,
        hue="Type",
        hue_order=["Coniferous", "Deciduous"],
        palette=["#2A5C03", "#DAA520"],
        multiple="stack",
        shrink=0.75,
        alpha=1,
        lw=0,
    )

    ax.set_ylim(gdf["plot"].max() + 0.5, gdf["plot"].min() - 0.5)
    ax.set_ylabel("Plot", fontsize=12)
    ax.set_xlabel("Tree Count", fontsize=12)
    ax.set_title("Tree Type Distribution by Plot", fontsize=16)
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(50))
    ax.grid(axis="x", color="black", alpha=0.1)
    ax.grid(axis="x", which="minor", color="black", alpha=0.1)

    save_plot(save_path)


def plot_species_distribution_for_all_plots(gdf, output_dir):
    """
    Plots the spatial distribution of species for all plots and saves each plot with a corresponding name.

    Parameters:
    - gdf (GeoDataFrame): The GeoDataFrame containing tree data.
    - output_dir (str): Directory to save the plot images.
    """
    logger.info("Visualizing species distribution for all plots...")

    os.makedirs(output_dir, exist_ok=True)

    unique_plots = gdf["plot"].unique()
    for plot_number in unique_plots:
        logger.info(f"Processing plot {plot_number}...")

        plot_data = gdf.query("plot == @plot_number")

        if plot_data.empty:
            logger.warning(f"No data available for plot {plot_number}. Skipping.")
            continue

        ax = plot_data.plot(
            column="species",
            legend=True,
            s=5,
            aspect="equal",
            figsize=(8, 8),
            cmap="viridis",
        )
        ax.set_title(f"Species Distribution in Plot {int(plot_number)}", fontsize=16)

        save_path = os.path.join(output_dir, f"species_distribution_plot_{int(plot_number)}.png")
        save_plot(save_path)


# --- Main Workflow ---
def process_data(data_dir, output_dir):
    """
    Main workflow for processing data.

    Parameters:
    - data_dir (str): Directory containing input data.
    - output_dir (str): Directory to save processed data and visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load and inspect GeoJSON
    field_survey_path = os.path.join(data_dir, "field_survey.geojson")
    field_survey = gpd.read_file(field_survey_path)
    inspect_geojson_data(field_survey, output_dir)
    plot_geojson_species_map(field_survey, save_path=os.path.join(output_dir, "species_map.png"))

    # Inspect LAS files
    inspect_las_data(os.path.join(data_dir, "als"), output_dir)
    plot_tree_type_distribution_by_plot(
        gdf=field_survey,
        save_path=os.path.join(output_dir, "tree_type_distribution_by_plot.png"),
    )

    plot_species_distribution_for_all_plots(
        gdf=field_survey,
        output_dir=os.path.join(output_dir, "species_distribution_plots"),
    )

    visualize_raster_images(os.path.join(data_dir, "ortho"), save_path=os.path.join(output_dir, "raster_images.png"))

    normalized_survey = normalize_geojson_heights(field_survey)
    normalized_survey.to_file(os.path.join(output_dir, "normalized_field_survey.geojson"), driver="GeoJSON")

    logger.info("Data processing complete!")