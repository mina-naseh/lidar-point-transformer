import matplotlib.pyplot as plt
import geopandas as gpd
import string
import logging
import seaborn as sns

# Use the global logging configuration from main
logger = logging.getLogger(__name__)

def plot_field_survey_map(gdf, species_col='species', save_path=None) -> None:
    """
    Plots a geographic map of trees colored by species and optionally saves the plot.

    Parameters:
    - gdf (GeoDataFrame): GeoDataFrame containing the field survey data with a 'geometry' column.
    - species_col (str): Column name representing tree species. Default is 'species'.
    - save_path (str, optional): Path to save the plot as an image file. Default is None.

    Returns:
    - None: Saves or displays the plot.
    """
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

def plot_individual_rectangles(gdf, plot_col='plot', save_path=None) -> None:
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
    unique_plots = sorted(gdf[plot_col].unique())
    plot_labels = {plot_id: f"Plot {letter}" for plot_id, letter in zip(unique_plots, string.ascii_uppercase)}

    fig, axes = plt.subplots(2, 5, figsize=(20, 10), sharex=False, sharey=False)
    axes = axes.flatten()

    for i, (plot_id, ax) in enumerate(zip(unique_plots, axes)):
        subset = gdf[gdf[plot_col] == plot_id]
        subset.plot(column='species', cmap='viridis', legend=False, ax=ax, markersize=9)
        ax.set_title(plot_labels[plot_id], fontsize=14)
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

def plot_species_counts(df, species_col, save_path=None) -> None:
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

    species_counts = df[species_col].value_counts().reset_index()
    species_counts.columns = [species_col, 'Count']

    plt.figure(figsize=(15, 6))
    sns.barplot(
        x='Count',
        y=species_col,
        data=species_counts,
        hue=species_col,
        palette="viridis"    
    )
    plt.legend([], [], frameon=False)
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

def plot_density(df, columns, save_path=None) -> None:
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
