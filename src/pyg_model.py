import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import PointTransformerConv, global_mean_pool

class PointTransformerNet(nn.Module):
    """
    A Point Transformer Network for binary classification of LiDAR points.

    Args:
        in_channels (int): Number of input features (e.g., 3 for x, y, z coordinates).
        out_channels (int): Number of output features (e.g., 1 for binary classification).
        hidden_channels (int): Number of hidden units for intermediate layers.
        pooling (bool): Whether to use global pooling (default: False).
    """
    def __init__(self, in_channels, out_channels, hidden_channels=64, pooling=False):
        super().__init__()
        self.pooling = pooling

        # Point Transformer layers
        self.conv1 = PointTransformerConv(in_channels, hidden_channels)
        self.conv2 = PointTransformerConv(hidden_channels, hidden_channels)

        # Fully connected layers for classification
        self.fc1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc2 = nn.Linear(hidden_channels // 2, out_channels)

    def forward(self, data):
        """
        Forward pass of the Point Transformer.

        Args:
            data (torch_geometric.data.Data): Input graph data with attributes:
                - x: Node features (N x F)
                - pos: Node positions (N x 3)
                - edge_index: Edge list (2 x E)
                - batch: Batch vector (N)

        Returns:
            torch.Tensor: Output predictions of shape (N, out_channels).
        """
        # Extract attributes from the PyG Data object
        x, pos, edge_index, batch = data.x, data.pos, data.edge_index, data.batch

        # Apply Point Transformer layers
        x = F.relu(self.conv1(x, pos, edge_index))
        x = F.relu(self.conv2(x, pos, edge_index))

        # Optional global pooling
        if self.pooling:
            x = global_mean_pool(x, batch)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # No activation, since BCEWithLogitsLoss expects raw logits

        return x
