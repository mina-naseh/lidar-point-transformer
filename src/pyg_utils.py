import numpy as np
import matplotlib.pyplot as plt


def filter_points_by_label(points, labels, target_label):
    """
    Filters points based on the target label.
    
    Parameters:
        points (np.ndarray): Array of point coordinates (N x 3).
        labels (np.ndarray): Array of point labels (N).
        target_label (int): Label to filter for.

    Returns:
        np.ndarray: Filtered points with the target label.
    """
    mask = labels == target_label
    return points[mask]


def normalize_coordinates(points):
    """
    Normalizes point cloud coordinates to zero mean and unit variance.
    
    Parameters:
        points (np.ndarray): Array of point coordinates (N x 3).

    Returns:
        np.ndarray: Normalized coordinates.
    """
    mean = np.mean(points, axis=0)
    std = np.std(points, axis=0)
    return (points - mean) / std


def visualize_point_cloud(points, labels=None, title="Point Cloud"):
    """
    Visualizes a point cloud in 3D.
    
    Parameters:
        points (np.ndarray): Array of point coordinates (N x 3).
        labels (np.ndarray): Optional array of labels for coloring the points (N).
        title (str): Title of the plot.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    if labels is not None:
        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=labels, cmap="tab10", s=1)
        plt.colorbar(scatter, ax=ax, label="Labels")
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
    ax.set_title(title)
    plt.show()

def compute_metrics(predictions, ground_truth):
    """
    Computes evaluation metrics for tree detection.
    
    Parameters:
        predictions (np.ndarray): Predicted labels (N).
        ground_truth (np.ndarray): Ground truth labels (N).

    Returns:
        dict: Dictionary with precision, recall, F1-score, and accuracy.
    """
    tp = np.sum((predictions == 1) & (ground_truth == 1))
    fp = np.sum((predictions == 1) & (ground_truth == 0))
    fn = np.sum((predictions == 0) & (ground_truth == 1))
    tn = np.sum((predictions == 0) & (ground_truth == 0))
    
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "accuracy": accuracy,
    }
