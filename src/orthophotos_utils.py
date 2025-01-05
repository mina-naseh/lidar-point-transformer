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
    tiff_files = [os.path.join(tif_dir, f'plot_{i:02d}.tif') for i in range(1, num_plots + 1)]
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, tiff_file in enumerate(tiff_files):
        with rasterio.open(tiff_file) as src:
            img = src.read(1)
            axes[i].imshow(img, cmap='gist_earth')
            axes[i].set_title(f'Plot {i + 1}')
            axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        logger.info(f"Raster images saved to {save_path}")
    else:
        plt.show()

    plt.close()