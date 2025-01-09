import os
import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
import logging
import laspy
import torch.nn.functional as F
import geopandas as gpd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_las_files(las_dir):
    """
    Load all .las files and extract points.
    """
    logger.info("Loading .las files...")
    las_files = [os.path.join(las_dir, f) for f in os.listdir(las_dir) if f.endswith(".las")]
    points_list = []
    for las_file in las_files:
        logger.info(f"Processing {las_file}...")
        las = laspy.read(las_file)
        mask = las.classification != 2  # Remove ground points
        points = np.stack([las.x[mask], las.y[mask], las.z[mask]], axis=1)
        points_list.append(points)
    logger.info(f"Loaded {len(points_list)} LAS files.")
    return points_list

def load_geojson(geojson_path):
    """
    Load tree inventory data from GeoJSON.
    """
    logger.info("Loading GeoJSON data...")
    geojson_data = gpd.read_file(geojson_path)
    logger.info(f"GeoJSON contains {len(geojson_data)} features.")
    return geojson_data

def match_trees_with_points(points_list, geojson_data, radius=1.0):
    """
    Match trees to points within a radius.
    """
    logger.info("Matching trees with points...")
    labels_list = []
    for i, points in enumerate(points_list):
        labels = np.zeros(len(points), dtype=np.float32)
        for _, tree in geojson_data.iterrows():
            tree_coords = np.array(tree.geometry.coords[0])  # Handle 2D coordinates
            distances = np.linalg.norm(points[:, :2] - tree_coords, axis=1)
            labels[distances < radius] = 1.0
        labels_list.append(labels)
        logger.info(f"Matched trees for LAS file {i + 1}/{len(points_list)}.")
    return labels_list

def prepare_data(las_dir, geojson_path, k=16, radius=1.0):
    """
    Prepare PyTorch Geometric Data objects for training.
    """

    points_list = load_las_files(las_dir)
    geojson_data = load_geojson(geojson_path)
    labels_list = match_trees_with_points(points_list, geojson_data, radius)

    # Find the maximum number of nodes across all graphs
    max_nodes = max(len(points) for points in points_list)
    pyg_data_list = []

    for idx, (points, labels) in enumerate(zip(points_list, labels_list)):
        coords = torch.tensor(points, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)

        nbrs = NearestNeighbors(n_neighbors=k).fit(coords)
        edges = nbrs.kneighbors_graph(coords, mode="connectivity").nonzero()
        edge_index = torch.tensor(np.array(edges)).T

        # Pad x and y tensors to ensure consistency across graphs
        pad_size = max_nodes - coords.size(0)
        if pad_size > 0:
            coords = F.pad(coords, (0, 0, 0, pad_size))  # Pad along the node dimension
            labels = F.pad(labels, (0, pad_size))

        # Debug edge index compatibility
        assert edge_index.max().item() < coords.size(0), f"Edge index out of bounds for Data {idx}"

        # Create a PyG Data object
        data = Data(x=coords, edge_index=edge_index, y=labels)
        pyg_data_list.append(data)

    return pyg_data_list


import os
import laspy
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import KNNGraph

def prepare_data_with_transform(las_dir, geojson_path, k=16, radius=1.0):
    """
    Prepares PyG Data objects with k-NN graph transform.
    
    Parameters:
        las_dir (str): Directory containing LAS files.
        geojson_path (str): Path to GeoJSON file for labeling.
        k (int): Number of neighbors for k-NN graph.
        radius (float): Radius for tree-label assignment.

    Returns:
        list: List of PyG Data objects.
    """
    points_list = load_las_files(las_dir)  # Your existing function to load LAS points
    geojson_data = load_geojson(geojson_path)  # Your function to load GeoJSON data
    labels_list = match_trees_with_points(points_list, geojson_data, radius)

    transform = KNNGraph(k=k)
    pyg_data_list = []

    for points, labels in zip(points_list, labels_list):
        # Extract 3D coordinates
        coords = torch.tensor(points, dtype=torch.float32)  # Shape: [N, 3]
        labels = torch.tensor(labels, dtype=torch.float32)  # Shape: [N]

        # Optional: Extract features (e.g., intensity) if available
        # Assuming `points` already includes features as additional columns
        features = coords.clone()  # Replace with actual features if available

        # Create Data object
        data = Data(x=features, pos=coords, y=labels)

        # Apply k-NN transform
        data = transform(data)

        # Append to the list
        pyg_data_list.append(data)

    return pyg_data_list
