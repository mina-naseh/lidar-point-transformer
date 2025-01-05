import numpy as np
import laspy
import logging
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import os
import glob

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Utility Functions ---

def load_las_file(file_path):
    """
    Loads a LAS file and returns its point cloud data.

    Parameters:
    - file_path (str): Path to the LAS file.

    Returns:
    - numpy.ndarray: Point cloud data (X, Y, Z).
    """
    try:
        las = laspy.read(file_path)
        points = np.vstack((las.x, las.y, las.z)).T
        logger.info(f"Loaded {points.shape[0]} points from {file_path}")
        return points
    except Exception as e:
        logger.error(f"Error loading LAS file {file_path}: {e}")
        return np.array([])


def filter_points_by_height(points, height_threshold=None, percentile=None):
    """
    Filters points based on a height threshold or a percentile of Z-values.

    Parameters:
    - points (numpy.ndarray): Point cloud data (X, Y, Z).
    - height_threshold (float, optional): Minimum Z value to retain points.
    - percentile (float, optional): Percentile of Z-values to compute threshold. If provided, overrides height_threshold.

    Returns:
    - numpy.ndarray: Filtered point cloud data.
    """
    if percentile is not None:
        z_threshold = np.percentile(points[:, 2], percentile)
        logger.info(f"Using percentile-based height threshold: {z_threshold:.2f}")
    elif height_threshold is not None:
        z_threshold = height_threshold
        logger.info(f"Using fixed height threshold: {z_threshold:.2f}")
    else:
        logger.warning("No height threshold or percentile provided. Returning all points.")
        return points

    filtered_points = points[points[:, 2] >= z_threshold]
    logger.info(f"Filtered points: {filtered_points.shape[0]} retained out of {points.shape[0]}")
    return filtered_points


def remove_noise_with_dbscan(points, eps=1.0, min_samples=5):
    """
    Removes noise from point clouds using DBSCAN clustering.

    Parameters:
    - points (numpy.ndarray): Point cloud data (X, Y, Z).
    - eps (float): The maximum distance between two samples for them to be in the same cluster.
    - min_samples (int): The minimum number of points required to form a dense region.

    Returns:
    - numpy.ndarray: Noise-free point cloud data.
    """
    if points.size == 0:
        logger.warning("No points provided for DBSCAN. Returning empty array.")
        return points

    try:
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    except ValueError as e:
        logger.error(f"DBSCAN clustering failed: {e}")
        return points

    labels = clustering.labels_
    filtered_points = points[labels >= 0]
    logger.info(f"DBSCAN retained {filtered_points.shape[0]} points out of {points.shape[0]}")
    return filtered_points

# --- Visualization Functions ---

def visualize_point_cloud(points, title="Point Cloud", save_path=None):
    """
    Visualizes a 3D point cloud.

    Parameters:
    - points (numpy.ndarray): Point cloud data (X, Y, Z).
    - title (str): Title for the plot. Default is 'Point Cloud'.
    - save_path (str, optional): Path to save the plot. Default is None.

    Returns:
    - None: Displays or saves the plot.
    """
    if points.size == 0:
        logger.warning("No points to visualize.")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], s=1, cmap="viridis")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title(title)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        logger.info(f"{title} plot saved to {save_path}")
    else:
        plt.show()

    plt.close()

def visualize_filtered_point_cloud(file_path, height_threshold=None, apply_dbscan=False, eps=1.0, min_samples=5, save_path=None):
    """
    Visualizes a filtered 3D point cloud from a LAS file.

    Parameters:
    - file_path (str): Path to the LAS file.
    - height_threshold (float, optional): Minimum height for filtering. Default is None.
    - apply_dbscan (bool): Whether to apply DBSCAN clustering. Default is False.
    - eps (float): Maximum distance for DBSCAN clustering. Default is 1.0.
    - min_samples (int): Minimum samples for DBSCAN clustering. Default is 5.
    - save_path (str, optional): Path to save the plot. Default is None.

    Returns:
    - None: Displays or saves the plot.
    """
    points = load_las_file(file_path)
    if points.size == 0:
        logger.warning("No points to visualize.")
        return

    if height_threshold is not None:
        points = filter_points_by_height(points, height_threshold)

    if apply_dbscan:
        points = remove_noise_with_dbscan(points, eps, min_samples)

    visualize_point_cloud(points, title=f"Filtered Point Cloud - {os.path.basename(file_path)}", save_path=save_path)

def process_and_visualize_multiple_point_clouds(
    las_dir, save_dir=None, apply_dbscan=False, eps=1.0, min_samples=5, percentile=None
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

    Returns:
    - None: Displays or saves the plots.
    """
    # Collect all LAS files from the directory
    las_files = [os.path.join(las_dir, f) for f in os.listdir(las_dir) if f.endswith(".las")]
    if not las_files:
        logger.warning(f"No LAS files found in directory: {las_dir}")
        return

    # Process each LAS file
    for idx, file_path in enumerate(las_files, start=1):
        points = load_las_file(file_path)
        if points.size == 0:
            logger.warning(f"File {file_path} is empty. Skipping...")
            continue

        # Apply percentile-based Z-value threshold if specified
        if percentile is not None:
            z_threshold = np.percentile(points[:, 2], percentile)
            logger.info(f"Using {percentile}th percentile as Z-value threshold: {z_threshold:.3f}")
            points = filter_points_by_height(points, z_threshold)

        # Apply DBSCAN clustering if enabled
        if apply_dbscan:
            points = remove_noise_with_dbscan(points, eps=eps, min_samples=min_samples)

        # Visualize and optionally save the filtered point cloud
        plot_title = f"Filtered Point Cloud - {os.path.basename(file_path)}"
        save_path = os.path.join(save_dir, f"plot_{idx}.png") if save_dir else None
        visualize_point_cloud(points, title=plot_title, save_path=save_path)
