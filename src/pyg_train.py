import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from .pyg_model import PointTransformerNet
from .pyg_utils import compute_metrics
import logging

# Configure logging
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

    # Select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model
    model = PointTransformerNet(in_channels, out_channels, hidden_channels=64).to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    # Compute class imbalance ratio
    num_tree_points = sum(data.y.sum().item() for data in train_data)
    num_non_tree_points = sum((data.y == 0).sum().item() for data in train_data)
    pos_weight = num_non_tree_points / num_tree_points
    logger.info(f"Class Imbalance: Non-Tree Points / Tree Points = {pos_weight:.2f}")

    # Weighted loss function
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device))

    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            # Forward pass
            output = model(data).squeeze(-1)  # Output shape: [N]
            loss = criterion(output, data.y.float())  # Weighted binary classification loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    logger.info("Training complete.")
    return model, test_loader, device




from sklearn.metrics import precision_score, recall_score, f1_score

@torch.no_grad()
def test_model_per_file(model, test_loader, device):
    """
    Evaluates the trained model and calculates metrics per test file.

    Args:
        model (torch.nn.Module): Trained model.
        test_loader (DataLoader): DataLoader for test data.
        device (torch.device): Device used for computation.

    Returns:
        dict: Dictionary containing metrics (Accuracy, Precision, Recall, F1-Score) per file.
    """
    model.eval()
    per_file_metrics = {}

    for idx, data in enumerate(test_loader):
        data = data.to(device)
        output = model(data).squeeze(-1)  # Remove extra dimension
        preds = (output > 0).long()  # Threshold for binary classification

        # Flatten predictions and labels
        preds = preds.cpu().numpy()
        labels = data.y.cpu().numpy()

        # Compute metrics for the current file
        accuracy = (preds == labels).mean()
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)

        # Store metrics for the current file
        per_file_metrics[f"File {idx + 1}"] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
        }

    print("Per-file metrics:")
    for file_name, metrics in per_file_metrics.items():
        print(f"{file_name}: {metrics}")

    return per_file_metrics
