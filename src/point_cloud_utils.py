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

LAS_GROUND_CLASS = 2  # Classification code for ground points in LAS files

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





# --- LMF Functions ---
def normalize_cloud_height(points, ground_points):
    """
    Normalizes the Z-values of the point cloud relative to the ground level.

    Parameters:
    - points (numpy.ndarray): Full point cloud data (X, Y, Z).
    - ground_points (numpy.ndarray): Ground points data (X, Y, Z).

    Returns:
    - numpy.ndarray: Point cloud with normalized heights (X, Y, Z).
    """
    logger.info("Normalizing heights...")
    ground_level = griddata(
        points=ground_points[:, :2],
        values=ground_points[:, 2],
        xi=points[:, :2],
        method="nearest"
    )
    normalized_points = points.copy()
    normalized_points[:, 2] -= ground_level  # Subtract ground level to normalize height
    logger.info("Height normalization complete.")
    return normalized_points


def local_maxima_filter(cloud, window_size, height_threshold):
    """
    Detects local maxima in the point cloud with a fixed window size.

    Parameters:
    - cloud (numpy.ndarray): Point cloud data (X, Y, Z).
    - window_size (float): Radius of the neighborhood to consider for local maxima.
    - height_threshold (float): Minimum height threshold for local maxima detection.

    Returns:
    - numpy.ndarray: Detected tree locations and heights.
    """
    logger.info(f"Applying Local Maxima Filtering with window_size={window_size}, height_threshold={height_threshold}")
    cloud = cloud[cloud[:, 2] > height_threshold]
    tree = scipy.spatial.KDTree(data=cloud)
    seen_mask = np.zeros(cloud.shape[0], dtype=bool)
    local_maxima = []

    for i, point in enumerate(cloud):
        if seen_mask[i]:
            continue
        neighbor_indices = tree.query_ball_point(point, window_size)
        highest_neighbor = neighbor_indices[cloud[neighbor_indices, 2].argmax()]
        seen_mask[neighbor_indices] = True
        seen_mask[highest_neighbor] = False

        if i == highest_neighbor:
            local_maxima.append(i)

    logger.info(f"Detected {len(local_maxima)} trees.")
    return cloud[local_maxima]


def process_point_cloud_with_lmf(points, ground_points, window_size, height_threshold):
    """
    Processes a point cloud with height normalization and local maxima filtering.

    Parameters:
    - points (numpy.ndarray): Full point cloud data (X, Y, Z).
    - ground_points (numpy.ndarray): Ground points data (X, Y, Z).
    - window_size (float): Radius for local maxima detection.
    - height_threshold (float): Minimum height threshold for local maxima detection.

    Returns:
    - numpy.ndarray: Detected tree locations and heights.
    """
    # Normalize heights
    normalized_points = normalize_cloud_height(points, ground_points)

    # Apply Local Maxima Filtering
    detected_trees = local_maxima_filter(normalized_points, window_size, height_threshold)

    return detected_trees


# --- Updated Processing Pipeline ---
def process_and_visualize_multiple_point_clouds_with_lmf(
    las_dir, save_dir=None, apply_dbscan=False, eps=1.0, min_samples=5,
    window_size=2.0, height_threshold=3.0, percentile=None
):
    """
    Processes and visualizes multiple LAS files with optional DBSCAN and LMF.

    Parameters:
    - las_dir (str): Directory containing LAS files.
    - save_dir (str, optional): Directory to save plots and results.
    - apply_dbscan (bool): Whether to apply DBSCAN clustering.
    - eps (float): Maximum distance for DBSCAN clustering.
    - min_samples (int): Minimum samples for DBSCAN clustering.
    - window_size (float): Radius for local maxima detection.
    - height_threshold (float): Minimum height threshold for LMF.
    - percentile (float, optional): Percentile of Z-values to compute threshold.

    Returns:
    - None: Displays or saves the plots and results.
    """
    las_files = [os.path.join(las_dir, f) for f in os.listdir(las_dir) if f.endswith(".las")]
    if not las_files:
        logger.warning(f"No LAS files found in directory: {las_dir}")
        return

    for idx, file_path in enumerate(las_files, start=1):
        logger.info(f"Processing {os.path.basename(file_path)}...")
        las = laspy.read(file_path)

        # Separate ground and non-ground points
        ground_points = las.xyz[las.classification == LAS_GROUND_CLASS]
        non_ground_points = las.xyz[las.classification != LAS_GROUND_CLASS]

        if non_ground_points.size == 0 or ground_points.size == 0:
            logger.warning(f"File {file_path} is missing required data. Skipping...")
            continue

        # Optional DBSCAN noise removal
        if apply_dbscan:
            non_ground_points = remove_noise_with_dbscan(non_ground_points, eps=eps, min_samples=min_samples)

        # Apply Local Maxima Filtering (LMF)
        detected_trees = process_point_cloud_with_lmf(
            points=non_ground_points,
            ground_points=ground_points,
            window_size=window_size,
            height_threshold=height_threshold
        )

        # Save detected tree locations as GeoJSON
        detected_trees_gdf = gpd.GeoDataFrame(
            data={"height": detected_trees[:, 2]},
            geometry=gpd.points_from_xy(detected_trees[:, 0], detected_trees[:, 1], crs="EPSG:32640")
        )
        if save_dir:
            geojson_path = os.path.join(save_dir, f"detected_trees_{idx}_lmf.geojson")
            detected_trees_gdf.to_file(geojson_path, driver="GeoJSON")
            logger.info(f"Detected trees saved to {geojson_path}")

        # Visualize the filtered point cloud and detected trees
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(non_ground_points[:, 0], non_ground_points[:, 1], c=non_ground_points[:, 2], s=1, cmap="viridis", label="Point Cloud")
        ax.scatter(detected_trees[:, 0], detected_trees[:, 1], color="red", s=10, label="Detected Trees")
        ax.set_title(f"Point Cloud with Detected Trees - {os.path.basename(file_path)}")
        ax.legend()
        plt.tight_layout()

        if save_dir:
            plot_path = os.path.join(save_dir, f"plot_{idx}_lmf.png")
            plt.savefig(plot_path, bbox_inches="tight")
            logger.info(f"Plot saved to {plot_path}")
        else:
            plt.show()
        plt.close()


# --- checks ---

def match_candidates(
    ground_truth: np.ndarray,
    candidates: np.ndarray,
    max_distance: float,
    max_height_difference: float,
) -> list[dict]:
    """Match ground truth trees to candidates."""
    distance_matrix = cdist(ground_truth[:, :2], candidates[:, :2])
    matches = []

    for gt_idx, gt_point in enumerate(ground_truth):
        closest_idx = np.argmin(distance_matrix[gt_idx])
        distance = distance_matrix[gt_idx, closest_idx]
        height_diff = abs(gt_point[2] - candidates[closest_idx, 2])
        if distance <= max_distance and height_diff <= max_height_difference:
            matches.append({"ground_truth": gt_point, "candidate": candidates[closest_idx], "distance": distance})
        else:
            matches.append({"ground_truth": gt_point, "candidate": None, "distance": None})

    for cand_idx, cand_point in enumerate(candidates):
        if not any(m["candidate"] is not None and np.array_equal(m["candidate"], cand_point) for m in matches):
            matches.append({"ground_truth": None, "candidate": cand_point, "distance": None})

    return matches


def visualize_detection_results(
    detected_trees, ground_truth, matches, save_path=None
):
    """
    Visualizes detected trees, ground truth, and matches.

    Parameters:
    - detected_trees (np.ndarray): Detected tree locations and heights.
    - ground_truth (np.ndarray): Ground truth tree locations and heights.
    - matches (list): Match results from match_candidates.
    - save_path (str, optional): Path to save the plot.

    Returns:
    - None: Displays or saves the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(ground_truth[:, 0], ground_truth[:, 1], c="green", label="Ground Truth", s=50, alpha=0.7)
    ax.scatter(detected_trees[:, 0], detected_trees[:, 1], c="red", label="Detected Trees", s=50, alpha=0.7)

    for match in matches:
        if match["distance"] is not None:
            ax.plot(
                [match["ground_truth"][0], match["candidate"][0]],
                [match["ground_truth"][1], match["candidate"][1]],
                c="blue", linestyle="--", alpha=0.5,
            )

    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.legend()
    plt.title("Tree Detection Results")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        logger.info(f"Detection results plot saved to {save_path}")
    else:
        plt.show()

def calculate_detection_metrics(matches):
    """
    Calculates detection metrics (precision, recall, F1-score) from match results.

    Parameters:
    - matches (list): Match results from match_candidates.

    Returns:
    - dict: Precision, recall, F1-score, and mean distance.
    """
    tp = sum(1 for m in matches if m["ground_truth"] is not None and m["candidate"] is not None)
    fp = sum(1 for m in matches if m["ground_truth"] is None and m["candidate"] is not None)
    fn = sum(1 for m in matches if m["ground_truth"] is not None and m["candidate"] is None)

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    mean_distance = np.mean([m["distance"] for m in matches if m["distance"] is not None])

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mean_distance": mean_distance,
    }



def process_all_las_files_with_ground_truth(
    las_dir,
    ground_truth_data,
    save_dir,
    max_distance=5.0,
    max_height_difference=3.0,
    window_size=2.0,
    height_threshold=3.0,
):
    """
    Process all LAS files in the given directory, match detected trees with ground truth, 
    and calculate metrics.

    Parameters:
    - las_dir (str): Directory containing LAS files.
    - ground_truth_data (GeoDataFrame): Ground truth data.
    - save_dir (str): Directory to save results and visualizations.
    - max_distance (float): Maximum distance for matching candidates.
    - max_height_difference (float): Maximum height difference for matching.
    - window_size (float): Window size for Local Maxima Filtering.
    - height_threshold (float): Height threshold for Local Maxima Filtering.

    Returns:
    - pd.DataFrame: Summary of detection metrics for all plots.
    """
    las_files = sorted(glob.glob(os.path.join(las_dir, "*.las")))
    if not las_files:
        logger.warning(f"No LAS files found in directory: {las_dir}")
        return pd.DataFrame()

    metrics_list = []

    for idx, las_file in enumerate(las_files, start=1):
        logger.info(f"Processing {os.path.basename(las_file)}...")

        # Load the LAS file
        points = load_las_file(las_file)  # Load the full point cloud (X, Y, Z, classification)

        # Split points into ground and non-ground points based on classification
        ground_points = points[points[:, -1] == LAS_GROUND_CLASS]  # Assuming classification is in the last column
        non_ground_points = points[points[:, -1] != LAS_GROUND_CLASS]

        # Process the LAS file using `process_point_cloud_with_lmf`
        detected_trees = process_point_cloud_with_lmf(
            points=non_ground_points,
            ground_points=ground_points,
            window_size=window_size,
            height_threshold=height_threshold,
        )


        # Extract ground truth for this plot
        plot_ground_truth = ground_truth_data[ground_truth_data["plot"] == idx]
        if not plot_ground_truth.empty:
            plot_ground_truth["geometry.x"] = plot_ground_truth.geometry.x
            plot_ground_truth["geometry.y"] = plot_ground_truth.geometry.y
            plot_ground_truth = plot_ground_truth[["geometry.x", "geometry.y", "height"]].to_numpy()
        else:
            logger.warning(f"No ground truth data found for plot {idx}.")
            continue



        # Match detected trees with ground truth
        matches = match_candidates(
            ground_truth=plot_ground_truth,
            candidates=detected_trees,
            max_distance=max_distance,
            max_height_difference=max_height_difference,
        )

        # Calculate metrics
        metrics = calculate_detection_metrics(matches)
        metrics["plot"] = idx
        metrics_list.append(metrics)

        # Visualize detection results
        save_path = os.path.join(save_dir, f"plot_{idx}_detection_results.png")
        visualize_detection_results(detected_trees, plot_ground_truth, matches, save_path=save_path)

    # Summarize and save metrics
    metrics_summary = pd.DataFrame(metrics_list)
    metrics_summary.to_csv(os.path.join(save_dir, "detection_metrics.csv"), index=False)
    logger.info("Detection metrics summary saved.")

    return metrics_summary
