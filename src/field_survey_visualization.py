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

def plot_geojson_species_map(gdf, species_col='species', save_path=None) -> None:
    """
    Plots a geographic map of trees colored by species and optionally saves the plot.

    Parameters:
    - gdf (GeoDataFrame): GeoDataFrame containing the field survey data with a 'geometry' column.
    - species_col (str): Column name representing tree species. Default is 'species'.
    - save_path (str, optional): Path to save the plot as an image file. Default is None.

    Returns:
    - None: Saves or displays the plot.
    """
    if gdf.empty:
        logger.warning("GeoDataFrame is empty. Skipping geographic map plotting.")
        return

    try:
        gdf = gdf.to_crs(epsg=4326)  # Convert to WGS84
        ax = gdf.plot(column=species_col, cmap='viridis', legend=True, figsize=(15, 15), markersize=1)
        plt.title('Geographic plot of trees, colored by species', fontsize=21)
        
        legend = ax.get_legend()
        if legend:
            for label in legend.get_texts():
                label.set_fontsize(16)
        
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"Field survey map saved to {save_path}")
        else:
            plt.show()

        plt.close()
    except Exception as e:
        logger.error(f"Error during species map plotting: {e}")

def plot_field_survey_subplots(gdf, plot_col='plot', save_path=None) -> None:
    """
    Creates a subplot for each unique plot ID, showing the geographic distribution of trees,
    and optionally saves the combined plot.

    Parameters:
    - gdf (GeoDataFrame): GeoDataFrame containing the field survey data with a 'geometry' column.
    - plot_col (str): Column name representing plot IDs. Default is 'plot'.
    - save_path (str, optional): Path to save the combined plot as an image file. Default is None.

    Returns:
    - None: Saves or displays the plot.
    """
    if gdf.empty:
        logger.warning("GeoDataFrame is empty. Skipping individual rectangles plotting.")
        return

    try:
        unique_plots = sorted(gdf[plot_col].unique())
        logger.info(f"Creating subplots for {len(unique_plots)} unique plots.")

        # Determine subplot grid size
        num_cols = 5
        num_rows = math.ceil(len(unique_plots) / num_cols)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 5), sharex=False, sharey=False)
        axes = axes.flatten()

        for i, (plot_id, ax) in enumerate(zip(unique_plots, axes)):
            subset = gdf[gdf[plot_col] == plot_id]
            subset.plot(column='species', cmap='viridis', legend=False, ax=ax, markersize=9)
            ax.set_title(f"Plot {plot_id}", fontsize=14)
            ax.set_xlim(subset.total_bounds[[0, 2]])
            ax.set_ylim(subset.total_bounds[[1, 3]])

        for ax in axes[len(unique_plots):]:
            ax.set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"Individual rectangles plot saved to {save_path}")
        else:
            plt.show()

        plt.close()
    except Exception as e:
        logger.error(f"Error during subplot plotting: {e}")

def plot_species_bar_chart(df, species_col='species', save_path=None) -> None:
    """
    Creates a bar plot for species counts.

    Parameters:
    - df (DataFrame): DataFrame containing the species column.
    - species_col (str): Name of the column with species data.
    - save_path (str, optional): Path to save the plot. Default is None.

    Returns:
    - None: Saves or displays the plot.
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
            hue=species_col,  # Set hue to the same as y
            dodge=False,      # Disable dodging for the bar
            palette="viridis"
        )
        plt.legend([], [], frameon=False)  # Remove legend manually

        plt.title("Species Count", fontsize=16)
        plt.xlabel("Count", fontsize=12)
        plt.ylabel("Species", fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            logger.info(f"Species count plot saved to {save_path}")
        else:
            plt.show()

        plt.close()
    except Exception as e:
        logger.error(f"Error during species bar chart plotting: {e}")

def plot_field_density(df, columns, save_path=None) -> None:
    """
    Creates density plots for specified columns.

    Parameters:
    - df (DataFrame): DataFrame containing the columns to plot.
    - columns (list): List of column names to plot density distributions for.
    - save_path (str, optional): Path to save the combined plot. Default is None.

    Returns:
    - None: Saves or displays the plot.
    """
    if df.empty:
        logger.warning("The DataFrame is empty. Skipping density plot.")
        return

    try:
        logger.info(f"Creating density plots for {len(columns)} columns: {columns}.")
        plt.figure(figsize=(15, len(columns) * 3))
        for i, col in enumerate(columns, 1):
            plt.subplot(1, len(columns), i)
            sns.kdeplot(df[col], fill=True)
            plt.title(f"Density plot for {col}", fontsize=14)
            plt.xlabel(col, fontsize=12)
            plt.ylabel("Density", fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            logger.info(f"Density plots saved to {save_path}")
        else:
            plt.show()

        plt.close()
    except Exception as e:
        logger.error(f"Error during density plot creation: {e}")


def visualize_raster_images(tif_dir, num_plots=10, save_path=None):
    """
    Visualizes raster images from a specified directory.

    Parameters:
    - tif_dir (str): Path to the directory containing raster files (.tif).
    - num_plots (int): Number of plots to visualize. Default is 10.
    - save_path (str, optional): Path to save the combined plot as an image file. Default is None.

    Returns:
    - None: Saves or displays the raster visualization.
    """
    if not os.path.exists(tif_dir):
        logger.error(f"Directory does not exist: {tif_dir}")
        return

    tiff_files = [
        os.path.join(tif_dir, f'plot_{i:02d}.tif') for i in range(1, num_plots + 1)
    ]
    tiff_files = [f for f in tiff_files if os.path.exists(f)]  # Filter missing files

    if not tiff_files:
        logger.warning(f"No valid raster files found in directory: {tif_dir}")
        return

    logger.info(f"Visualizing {len(tiff_files)} raster images from {tif_dir}")

    # Determine subplot grid dynamically
    num_rows = (len(tiff_files) + 4) // 5  # Ensure enough rows for up to 5 columns
    fig, axes = plt.subplots(num_rows, 5, figsize=(20, num_rows * 5))
    axes = axes.flatten()

    for i, tiff_file in enumerate(tiff_files):
        try:
            with rasterio.open(tiff_file) as src:
                img = src.read(1)
                axes[i].imshow(img, cmap='gist_earth')
                axes[i].set_title(os.path.basename(tiff_file).split('.')[0], fontsize=12)
                axes[i].axis('off')
        except Exception as e:
            logger.warning(f"Error reading raster file {tiff_file}: {e}")
            axes[i].set_visible(False)  # Hide the subplot for this file

    # Hide any unused axes
    for ax in axes[len(tiff_files):]:
        ax.set_visible(False)

    plt.tight_layout()

    if save_path:
        try:
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"Raster images saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save raster images: {e}")
    else:
        plt.show()

    plt.close()