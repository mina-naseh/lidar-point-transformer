import numpy as np
import matplotlib.pyplot as plt

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
