import os
import matplotlib.pyplot as plt
import rasterio
import logging

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

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
