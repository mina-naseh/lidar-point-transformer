import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from .pyg_model import PointTransformerNet
from .pyg_utils import compute_metrics
import logging
from src.pyg_utils import visualize_point_cloud, plot_training_metrics
from sklearn.metrics import precision_score, recall_score, f1_score
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train_model(train_data, test_data, in_channels, out_channels, epochs, batch_size, lr):
    """
    Trains the Point Transformer model with weighted loss.

    Args:
        train_data (list): List of PyG Data objects for training.
        test_data (list): List of PyG Data objects for testing.
        in_channels (int): Number of input features per point (e.g., 3 for x, y, z).
        out_channels (int): Number of output features per point (1 for binary classification).
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for DataLoader.
        lr (float): Learning rate for the optimizer.

    Returns:
        tuple: (model, test_loader, device) - Trained model, test DataLoader, and device used.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = PointTransformerNet(in_channels, out_channels, hidden_channels=64).to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    num_tree_points = sum(data.y.sum().item() for data in train_data)
    num_non_tree_points = sum((data.y == 0).sum().item() for data in train_data)
    pos_weight = num_non_tree_points / num_tree_points
    logger.info(f"Class Imbalance: Non-Tree Points / Tree Points = {pos_weight:.2f}")

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    metrics = {"loss": [], "precision": [], "recall": [], "f1_score": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            output = model(data).squeeze(-1)
            loss = criterion(output, data.y.float())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = (output > 0).long().cpu()
            all_preds.append(preds)
            all_labels.append(data.y.cpu())

        avg_loss = total_loss / len(train_loader)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        metrics["loss"].append(avg_loss)
        metrics["precision"].append(precision)
        metrics["recall"].append(recall)
        metrics["f1_score"].append(f1)

        logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

    plot_training_metrics(metrics, save_dir="./results_pyg")

    return model, test_loader, device


@torch.no_grad()
def test_model_per_file_with_visualization(model, test_loader, device, save_dir="./results_pyg"):
    """
    Evaluates the trained model, calculates metrics per test file, and saves visualizations.

    Args:
        model (torch.nn.Module): Trained model.
        test_loader (DataLoader): DataLoader for test data.
        device (torch.device): Device used for computation.
        save_dir (str): Directory to save plots.

    Returns:
        dict: Dictionary containing metrics (Accuracy, Precision, Recall, F1-Score) per file.
    """
    model.eval()
    per_file_metrics = {}
    os.makedirs(save_dir, exist_ok=True)

    for idx, data in enumerate(test_loader):
        data = data.to(device)
        output = model(data).squeeze(-1)
        preds = (output > 0).long()

        preds = preds.cpu()
        labels = data.y.cpu()

        accuracy = (preds == labels).float().mean().item()
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)

        per_file_metrics[f"File {idx + 1}"] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
        }

        save_path = os.path.join(save_dir, f"point_cloud_file_{idx + 1}.png")
        visualize_point_cloud(data, predictions=preds, title=f"Test File {idx + 1}", save_path=save_path)

    return per_file_metrics
