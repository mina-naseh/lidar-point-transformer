import os
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.transforms import KNNGraph
import logging
import geopandas as gpd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_npy_files(npy_dir):
    """
    Load preprocessed .npy files containing vegetation points from subdirectories.
    """
    logger.info("Loading preprocessed .npy files from subdirectories...")
    points_list = []

    for root, _, files in os.walk(npy_dir):  # Recursively traverse subdirectories
        for file in files:
            if file.endswith("_vegetation.npy"):  # Only load vegetation files
                file_path = os.path.join(root, file)
                logger.info(f"Loading {file_path}...")
                points = np.load(file_path)
                points_list.append(points)

    logger.info(f"Loaded {len(points_list)} vegetation .npy files.")
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
        logger.info(f"Matched trees for .npy file {i + 1}/{len(points_list)}.")
    return labels_list

def prepare_data_with_transform(npy_dir, geojson_path, k=16, radius=1.0):
    """
    Prepares PyG Data objects with k-NN graph transform for Point Transformer.
    
    Parameters:
        npy_dir (str): Directory containing preprocessed .npy files.
        geojson_path (str): Path to GeoJSON file for labeling.
        k (int): Number of neighbors for k-NN graph.
        radius (float): Radius for tree-label assignment.

    Returns:
        list: List of PyG Data objects.
    """
    points_list = load_npy_files(npy_dir)  # Load preprocessed vegetation points from .npy files
    geojson_data = load_geojson(geojson_path)  # Load GeoJSON
    labels_list = match_trees_with_points(points_list, geojson_data, radius)

    transform = KNNGraph(k=k)  # PyG transform for k-NN graph construction
    pyg_data_list = []

    for points, labels in zip(points_list, labels_list):
        coords = torch.tensor(points, dtype=torch.float32)  # 3D coordinates
        labels = torch.tensor(labels, dtype=torch.float32)  # Binary labels

        # Normalize coordinates for better training stability
        coords = (coords - coords.mean(dim=0)) / coords.std(dim=0)

        # Optional: Extract additional features if available
        features = coords.clone()  # Replace with actual features if present

        # Create PyG Data object
        data = Data(x=features, pos=coords, y=labels)

        # Apply k-NN graph construction
        data = transform(data)

        # Append to the list
        pyg_data_list.append(data)
        logger.info(f"Prepared PyG Data object for one .npy file with {data.num_nodes} nodes.")

    logger.info(f"Prepared {len(pyg_data_list)} PyG Data objects.")
    return pyg_data_list
