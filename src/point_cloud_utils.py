import os
import matplotlib.pyplot as plt
import numpy as np
import laspy
import logging
from sklearn.cluster import DBSCAN

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


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


def filter_points_by_height(points, height_threshold):
    """
    Filters points based on a height threshold.
    """
    return points[points[:, 2] >= height_threshold]

def remove_noise_with_dbscan(points, eps=1.0, min_samples=5):
    """
    Removes noise from point clouds using DBSCAN clustering.
    """
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    return points[labels >= 0]
