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


def local_maxima_filter(cloud, window_size, height_threshold):
    """
    Detects local maxima in the point cloud with a fixed window size.

    Parameters:
    - cloud (numpy.ndarray): Point cloud data (X, Y, Z, classification).
    - window_size (float): Radius of the neighborhood to consider for local maxima.
    - height_threshold (float): Minimum height threshold for local maxima detection.

    Returns:
    - numpy.ndarray: Detected tree locations and heights (X, Y, Z, classification).
    """
    # Validate input
    if not isinstance(cloud, np.ndarray):
        raise TypeError(f"Cloud needs to be a numpy array, not {type(cloud)}")
    if cloud.size == 0:
        logger.warning("Point cloud is empty. Returning empty array.")
        return np.array([])

    # Filter points above the height threshold
    filtered_cloud = cloud[cloud[:, 2] > height_threshold]
    logger.info(f"Filtered {len(filtered_cloud)} points above height threshold {height_threshold}.")

    if filtered_cloud.size == 0:
        logger.warning("No points remain after height filtering. Returning empty array.")
        return np.array([])

    # Initialize KDTree for neighborhood queries (use full point: X, Y, Z)
    tree = scipy.spatial.KDTree(data=filtered_cloud)  # Full 3D KDTree (X, Y, Z)
    seen_mask = np.zeros(filtered_cloud.shape[0], dtype=bool)
    local_maxima = []

    # Detect local maxima
    for i, point in enumerate(filtered_cloud):
        if seen_mask[i]:
            continue

        # Find neighbors within the specified window size
        neighbor_indices = tree.query_ball_point(point, window_size)  # 3D spatial query
        if not neighbor_indices:
            continue

        # Find the index of the highest neighbor
        neighbor_heights = filtered_cloud[neighbor_indices, 2]
        highest_index = neighbor_indices[np.argmax(neighbor_heights)]

        # Mark all neighbors except the highest neighbor as seen
        for neighbor_index in neighbor_indices:
            if neighbor_index != highest_index:
                seen_mask[neighbor_index] = True

        # If the current point is the local maximum, record it
        if i == highest_index:
            local_maxima.append(i)

    logger.info(f"Detected {len(local_maxima)} local maxima (trees).")
    return filtered_cloud[local_maxima]


def process_point_cloud_with_lmf(points, ground_points, window_size, height_threshold):
    """
    Processes a point cloud with height normalization and local maxima filtering.

    Parameters:
    - points (numpy.ndarray): Full point cloud data (X, Y, Z, classification).
    - ground_points (numpy.ndarray): Ground points data (X, Y, Z).
    - window_size (float): Radius for local maxima detection.
    - height_threshold (float): Minimum height threshold for local maxima detection.

    Returns:
    - numpy.ndarray: Detected tree locations and heights (X, Y, Z, classification).
    """
    if points.size == 0:
        logger.warning("Point cloud is empty. Skipping processing.")
        return np.array([])

    if ground_points.size == 0:
        logger.warning("Ground points are empty. Returning original points.")
        return np.array([])

    # Normalize heights
    logger.info(f"Starting height normalization for {len(points)} points.")
    normalized_points = normalize_cloud_height(points, ground_points)
    logger.info(f"Normalized heights: min={normalized_points[:, 2].min()}, "
                f"max={normalized_points[:, 2].max()}, "
                f"mean={normalized_points[:, 2].mean()}")

    # Apply Local Maxima Filtering
    logger.info(f"Applying Local Maxima Filtering with window_size={window_size}, height_threshold={height_threshold}.")
    detected_trees = local_maxima_filter(normalized_points, window_size, height_threshold)
    logger.info(f"Local Maxima Filtering complete. Detected {len(detected_trees)} trees.")

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
    # Collect all LAS files from the directory
    las_files = [os.path.join(las_dir, f) for f in os.listdir(las_dir) if f.endswith(".las")]
    if not las_files:
        logger.warning(f"No LAS files found in directory: {las_dir}")
        return

    for file_path in sorted(las_files):
        try:
            plot_number = int(os.path.basename(file_path).split("_")[1].split(".")[0])
        except ValueError:
            logger.error(f"Could not extract plot number from file {file_path}. Skipping...")
            continue

        logger.info(f"Processing {os.path.basename(file_path)} (Plot {plot_number})...")

        # Load LAS file
        las = laspy.read(file_path)

        # Separate ground and non-ground points
        try:
            ground_points = las.xyz[las.classification == LAS_GROUND_CLASS]
            non_ground_points = las.xyz[las.classification != LAS_GROUND_CLASS]
        except AttributeError:
            logger.error(f"File {file_path} does not contain classification data. Skipping...")
            continue

        logger.info(f"Ground points: {len(ground_points)}, Non-ground points: {len(non_ground_points)}")

        if non_ground_points.size == 0 or ground_points.size == 0:
            logger.warning(f"No valid points found in {file_path}. Skipping...")
            continue

        # Apply height filtering (percentile or fixed threshold)
        non_ground_points = filter_points_by_height(non_ground_points, height_threshold=height_threshold, percentile=percentile)
        if non_ground_points.size == 0:
            logger.warning(f"No points remaining after height filtering for plot_{plot_number}. Skipping...")
            continue

        # Optional DBSCAN noise removal
        if apply_dbscan:
            original_count = len(non_ground_points)
            non_ground_points = remove_noise_with_dbscan(non_ground_points, eps=eps, min_samples=min_samples)
            # logger.info(f"DBSCAN retained {len(non_ground_points)} points out of {original_count}.")

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
            geojson_path = os.path.join(save_dir, f"detected_trees_plot_{plot_number}_lmf.geojson")
            detected_trees_gdf.to_file(geojson_path, driver="GeoJSON")
            logger.info(f"Detected trees saved to {geojson_path}")

        # Visualize the filtered point cloud and detected trees
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(non_ground_points[:, 0], non_ground_points[:, 1], c=non_ground_points[:, 2], s=1, cmap="viridis", label="Point Cloud")
        ax.scatter(detected_trees[:, 0], detected_trees[:, 1], color="red", s=10, label="Detected Trees")
        ax.set_title(f"Point Cloud with Detected Trees - Plot {plot_number}")
        ax.legend()
        plt.tight_layout()

        if save_dir:
            plot_path = os.path.join(save_dir, f"plot_{plot_number}_lmf.png")
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
    """
    Match ground truth trees to candidates.

    Parameters:
    - ground_truth (np.ndarray): Array of ground truth points (X, Y, Z).
    - candidates (np.ndarray): Array of candidate points (X, Y, Z).
    - max_distance (float): Maximum distance allowed for matching.
    - max_height_difference (float): Maximum height difference allowed for matching.

    Returns:
    - list[dict]: List of matches with ground truth, candidate, distance, and class (TP, FP, FN).
    """
    logger = logging.getLogger(__name__)

    # Handle empty inputs gracefully
    if ground_truth.size == 0:
        logger.warning("No ground truth points provided.")
        return [{"ground_truth": None, "candidate": tuple(cand), "class": "FP", "distance": None} for cand in candidates]
    if candidates.size == 0:
        logger.warning("No candidate points provided.")
        return [{"ground_truth": tuple(gt), "candidate": None, "class": "FN", "distance": None} for gt in ground_truth]

    logger.info(f"Matching {len(ground_truth)} ground truth trees with {len(candidates)} candidates.")

    # Compute distance matrix
    distance_matrix = scipy.spatial.distance_matrix(ground_truth[:, :2], candidates[:, :2])
    indices = np.nonzero(distance_matrix <= max_distance)
    distances = distance_matrix[indices]
    sparse_distances = sorted((d, pair) for d, pair in zip(distances, zip(*indices)))

    ground_truth_matched_mask = np.zeros(ground_truth.shape[0], dtype=bool)
    candidates_matched_mask = np.zeros(candidates.shape[0], dtype=bool)
    matches = []

    for distance, (i, j) in sparse_distances:
        if ground_truth_matched_mask[i] or candidates_matched_mask[j]:
            continue

        height_diff = abs(ground_truth[i, 2] - candidates[j, 2])
        if np.isnan(ground_truth[i, 2]) or height_diff <= max_height_difference:
            matches.append({
                "ground_truth": tuple(ground_truth[i]),
                "candidate": tuple(candidates[j]),
                "class": "TP",
                "distance": distance,
            })
            ground_truth_matched_mask[i] = True
            candidates_matched_mask[j] = True

    # Add unmatched ground truth (FN)
    matches.extend(
        {"ground_truth": tuple(ground_truth[i]), "candidate": None, "class": "FN", "distance": None}
        for i in range(len(ground_truth)) if not ground_truth_matched_mask[i]
    )

    # Add unmatched candidates (FP)
    matches.extend(
        {"ground_truth": None, "candidate": tuple(candidates[j]), "class": "FP", "distance": None}
        for j in range(len(candidates)) if not candidates_matched_mask[j]
    )

    logger.info(f"Matching complete. Total matches: {len(matches)}")
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
    if detected_trees.size == 0:
        logger.warning("No detected trees to visualize.")
        return
    if ground_truth.size == 0:
        logger.warning("No ground truth trees to visualize.")
        return
    if not matches:
        logger.warning("No matches to visualize.")
        return

    logger.info(f"Visualizing {len(ground_truth)} ground truth trees and {len(detected_trees)} detected trees.")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot for ground truth and detected trees
    ax.scatter(
        ground_truth[:, 0], ground_truth[:, 1],
        c="green", label="Ground Truth", s=50, alpha=0.7
    )
    ax.scatter(
        detected_trees[:, 0], detected_trees[:, 1],
        c="red", label="Detected Trees", s=50, alpha=0.7
    )

    # Plot matches with optional distance annotation
    for match in matches:
        if match["distance"] is not None:
            ax.plot(
                [match["ground_truth"][0], match["candidate"][0]],
                [match["ground_truth"][1], match["candidate"][1]],
                c="blue", linestyle="--", alpha=0.5,
            )
            # Optional: annotate with distance
            ax.text(
                (match["ground_truth"][0] + match["candidate"][0]) / 2,
                (match["ground_truth"][1] + match["candidate"][1]) / 2,
                f"{match['distance']:.2f}", fontsize=8, color="blue", alpha=0.7
            )

    # Adjust axis limits dynamically
    all_x = np.concatenate([ground_truth[:, 0], detected_trees[:, 0]])
    all_y = np.concatenate([ground_truth[:, 1], detected_trees[:, 1]])
    ax.set_xlim(all_x.min() - 10, all_x.max() + 10)
    ax.set_ylim(all_y.min() - 10, all_y.max() + 10)

    # Add labels and legend
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.legend()
    plt.title("Tree Detection Results")
    plt.tight_layout()

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        logger.info(f"Detection results plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def calculate_detection_metrics(matches):
    """
    Calculates detection metrics (precision, recall, F1-score) from match results.

    Parameters:
    - matches (list): Match results from match_candidates.

    Returns:
    - dict: Precision, recall, F1-score, and mean distance.
    """
    if not matches:
        logger.warning("No matches provided for metric calculation.")
        return {"precision": 0, "recall": 0, "f1_score": 0, "mean_distance": None}

    # Calculate true positives, false positives, and false negatives
    tp = sum(1 for m in matches if m["ground_truth"] is not None and m["candidate"] is not None)
    fp = sum(1 for m in matches if m["ground_truth"] is None and m["candidate"] is not None)
    fn = sum(1 for m in matches if m["ground_truth"] is not None and m["candidate"] is None)

    # Calculate metrics
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    mean_distance = (
        np.mean([m["distance"] for m in matches if m["distance"] is not None])
        if tp > 0 else None
    )

    # Log calculated metrics
    logger.info(f"Metrics calculated: TP={tp}, FP={fp}, FN={fn}")
    logger.info(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1-Score: {f1:.3f}")
    if mean_distance is not None:
        logger.info(f"Mean Distance: {mean_distance:.3f}")
    else:
        logger.info("Mean Distance: None (No true positives)")

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
    Processes all LAS files in a directory, matches detected trees with ground truth,
    and calculates detection metrics.

    Parameters:
    - las_dir (str): Directory containing LAS files.
    - ground_truth_data (GeoDataFrame): Ground truth data with tree locations.
    - save_dir (str): Directory to save results and visualizations.
    - max_distance (float): Maximum distance for matching candidates to ground truth.
    - max_height_difference (float): Maximum height difference for matching.
    - window_size (float): Window size for Local Maxima Filtering (LMF).
    - height_threshold (float): Height threshold for LMF.

    Returns:
    - pd.DataFrame: Summary of detection metrics for all plots.
    """
    las_files = sorted(glob.glob(os.path.join(las_dir, "*.las")))
    if not las_files:
        logger.warning(f"No LAS files found in directory: {las_dir}")
        return pd.DataFrame()

    metrics_list = []

    for las_file in las_files:
        try:
            # Extract plot number
            plot_number = int(os.path.basename(las_file).split("_")[1].split(".")[0])
        except (IndexError, ValueError):
            logger.error(f"Invalid file name format: {las_file}. Skipping...")
            continue

        logger.info(f"Processing {os.path.basename(las_file)} (Plot {plot_number})...")

        # Load LAS file
        points = load_las_file(las_file)
        if points.size == 0:
            logger.warning(f"No points in LAS file: {las_file}. Skipping...")
            continue

        # Split points based on classification
        ground_points = points[points[:, -1] == 2]  # Ground points
        non_ground_points = points[points[:, -1] == 5]  # High vegetation points

        logger.info(f"Number of ground points: {len(ground_points)}")
        logger.info(f"Number of high vegetation points: {len(non_ground_points)}")

        if non_ground_points.size == 0 or ground_points.size == 0:
            logger.warning(f"No valid points in {os.path.basename(las_file)}. Skipping...")
            continue

        # Process using Local Maxima Filtering
        detected_trees = process_point_cloud_with_lmf(
            points=non_ground_points,
            ground_points=ground_points,
            window_size=window_size,
            height_threshold=height_threshold,
        )
        logger.info(f"Detected {len(detected_trees)} trees using LMF.")

        # Match detected trees with ground truth
        plot_ground_truth = ground_truth_data[ground_truth_data["plot"] == plot_number].copy()

        if plot_ground_truth.empty:
            logger.warning(f"No ground truth data for plot {plot_number}. Skipping...")
            continue

        plot_ground_truth_np = transform_ground_truth(plot_ground_truth)

        matches = match_candidates(
            ground_truth=plot_ground_truth_np,
            candidates=detected_trees,
            max_distance=max_distance,
            max_height_difference=max_height_difference,
        )

        metrics = calculate_detection_metrics(matches)
        metrics["plot"] = plot_number
        metrics_list.append(metrics)

        # Visualize detection results
        save_path = os.path.join(save_dir, f"plot_{plot_number}_detection_results.png")
        visualize_detection_results(detected_trees, plot_ground_truth_np, matches, save_path=save_path)

    # Save and summarize metrics
    metrics_summary = pd.DataFrame(metrics_list)
    metrics_summary.to_csv(os.path.join(save_dir, "detection_metrics.csv"), index=False)
    logger.info("Detection metrics summary saved.")

    return metrics_summary


def transform_ground_truth(ground_truth):
    """
    Transforms ground truth GeoDataFrame to a NumPy array for processing.

    Parameters:
    - ground_truth (GeoDataFrame): Ground truth data.

    Returns:
    - np.ndarray: Transformed ground truth data as a NumPy array.
    """
    ground_truth["geometry.x"] = ground_truth.geometry.x
    ground_truth["geometry.y"] = ground_truth.geometry.y
    return ground_truth[["geometry.x", "geometry.y", "height"]].to_numpy()
