import os
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import laspy
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


def visualize_point_cloud(las_dir, num_plots=10, height_threshold=None, save_path=None):
    """
    Visualizes point cloud data from a specified directory.

    Parameters:
    - las_dir (str): Path to the directory containing LAS files.
    - num_plots (int): Number of plots to visualize. Default is 10.
    - height_threshold (float, optional): Minimum height threshold to filter points. Default is None.
    - save_path (str, optional): Path to save the combined plot as an image file. Default is None.

    Returns:
    - None: Saves or displays the point cloud visualization.
    """
    las_files = [f"{las_dir}/plot_{i:02d}.las" for i in range(1, num_plots + 1)]
    fig = plt.figure(figsize=(10, 20))

    for i, file in enumerate(las_files):
        las = laspy.read(file)
        points = np.vstack((las.x, las.y, las.z)).T

        # Normalize and filter by height if needed
        points -= points.min(axis=0, keepdims=True)
        if height_threshold:
            points = points[points[:, 2] >= height_threshold]

        ax = fig.add_subplot(5, 2, i + 1, projection="3d")
        scatter = ax.scatter(*points.T, c=points[:, 2], s=2, cmap="viridis")
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        ax.set_zlabel("Z (height in meters)")
        ax.set_title(f"Plot {i + 1}", y=0.85)
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.7, aspect=10, pad=0.1)
        cbar.set_label("Height (meters)")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        logger.info(f"Point cloud visualization saved to {save_path}")
    else:
        plt.show()

    plt.close()
