import os
import numpy as np
import laspy
import geopandas as gpd
import matplotlib.pyplot as plt
from src.pyg_utils import visualize_point_cloud, compute_metrics

# Paths to your data
LAS_DIR = "/Users/mina/Desktop/3rd_semester/wpI/lidar-point-transformer/data/als/"
GEOJSON_PATH = "/Users/mina/Desktop/3rd_semester/wpI/lidar-point-transformer/data/field_survey.geojson"


def load_las_files(las_dir):
    """
    Load all .las files and extract points.
    """
    print("Loading .las files...")
    las_files = [os.path.join(las_dir, f) for f in os.listdir(las_dir) if f.endswith(".las")]
    points_list = []
    for las_file in las_files:
        print(f"Processing {las_file}...")
        las = laspy.read(las_file)
        mask = las.classification != 2  # Remove ground points
        points = np.stack([las.x[mask], las.y[mask], las.z[mask]], axis=1)
        points_list.append(points)
    print(f"Loaded {len(points_list)} LAS files.")
    return points_list


def load_geojson(geojson_path):
    """
    Load tree inventory data from GeoJSON.
    """
    print("Loading GeoJSON data...")
    geojson_data = gpd.read_file(geojson_path)
    print(f"GeoJSON contains {len(geojson_data)} trees.")
    return geojson_data


def match_trees_with_points(points_list, geojson_data, radius=1.0):
    """
    Match trees to points within a radius.
    """
    print("Matching trees with points...")
    labels_list = []
    for i, points in enumerate(points_list):
        labels = np.zeros(len(points), dtype=np.float32)
        for _, tree in geojson_data.iterrows():
            tree_coords = np.array(tree.geometry.coords[0])  # Handle 2D coordinates
            distances = np.linalg.norm(points[:, :2] - tree_coords, axis=1)
            labels[distances < radius] = 1.0
        labels_list.append(labels)
        print(f"Matched trees for LAS file {i + 1}/{len(points_list)}.")
    return labels_list


def visualize_sample(points_list, labels_list):
    """
    Visualize a single point cloud sample with labels.
    """
    print("Visualizing a sample point cloud...")
    sample_idx = 0  # Change this to view different samples
    visualize_point_cloud(points_list[sample_idx], labels_list[sample_idx], title="Sample Point Cloud")


def test_pipeline():
    # Step 1: Load LiDAR and GeoJSON data
    points_list = load_las_files(LAS_DIR)
    geojson_data = load_geojson(GEOJSON_PATH)

    # Step 2: Match trees to LiDAR points
    labels_list = match_trees_with_points(points_list, geojson_data, radius=1.0)

    # Step 3: Visualize the data
    visualize_sample(points_list, labels_list)

    # Step 4: Compute basic statistics
    for i, (points, labels) in enumerate(zip(points_list, labels_list)):
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"File {i + 1}: Label Distribution: {dict(zip(unique_labels, counts))}")

    # Step 5: Basic metric testing
    predictions = np.random.choice([0, 1], size=len(labels_list[0]), p=[0.9, 0.1])  # Random predictions for testing
    metrics = compute_metrics(predictions, labels_list[0])
    print(f"Test Metrics for File 1: {metrics}")


if __name__ == "__main__":
    test_pipeline()
