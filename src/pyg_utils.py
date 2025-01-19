import numpy as np
import matplotlib.pyplot as plt
import os

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


def plot_training_metrics(metrics, save_dir="./results_pyg"):
    """
    Save training metrics plots.

    Args:
        metrics (dict): Dictionary containing lists of metrics over epochs (e.g., {"loss": [], "precision": []}).
        save_dir (str): Directory to save the plots.
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(metrics["loss"]) + 1)

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, metrics["loss"], label="Loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "training_loss.png"))
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, metrics["precision"], label="Precision", marker="o")
    plt.plot(epochs, metrics["recall"], label="Recall", marker="o")
    plt.plot(epochs, metrics["f1_score"], label="F1-Score", marker="o")
    plt.title("Training Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "training_metrics.png"))
    plt.close()


def visualize_point_cloud(data, predictions=None, title="Point Cloud", save_path="point_cloud.png"):
    """
    Save a 3D point cloud plot with ground truth labels and optional predictions.

    Args:
        data (torch_geometric.data.Data): PyG Data object with attributes:
            - pos: Point coordinates (N x 3)
            - y: Ground truth labels (N)
        predictions (torch.Tensor, optional): Predicted labels (N). Default: None.
        title (str): Title for the plot.
        save_path (str): Path to save the plot.
    """
    pos = data.pos.cpu().numpy()
    labels = data.y.cpu().numpy()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot ground truth
    ax.scatter(
        pos[labels == 1, 0], pos[labels == 1, 1], pos[labels == 1, 2], 
        c='purple', label="Ground Truth (Trees)", alpha=0.6, s=2
    )
    ax.scatter(
        pos[labels == 0, 0], pos[labels == 0, 1], pos[labels == 0, 2], 
        c='blue', label="Ground Truth (Non-Trees)", alpha=0.6, s=2
    )

    # Plot predictions if provided
    if predictions is not None:
        preds = predictions.cpu().numpy()
        ax.scatter(
            pos[preds == 1, 0], pos[preds == 1, 1], pos[preds == 1, 2], 
            c='yellow', label="Predictions (Trees)", alpha=0.3, s=1
        )
        ax.scatter(
            pos[preds == 0, 0], pos[preds == 0, 1], pos[preds == 0, 2], 
            c='cyan', label="Predictions (Non-Trees)", alpha=0.3, s=1
        )

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="upper right", fontsize=10)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)
