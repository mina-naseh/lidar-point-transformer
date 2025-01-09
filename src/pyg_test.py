import laspy
import geopandas as gpd
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define paths
las_dir = "./data/als/"  # Directory containing .las files
geojson_path = "./data/field_survey.geojson"  # Path to geojson file

# List all .las files
las_files = [os.path.join(las_dir, f) for f in os.listdir(las_dir) if f.endswith(".las")]

# Function to filter and summarize a LAS file
def process_las_file(las_path):
    las = laspy.read(las_path)
    unique_labels = set(las.classification)
    label_counts = {label: (las.classification == label).sum() for label in unique_labels}
    
    filtered_mask = las.classification != 2  # Keep only non-ground points (exclude label 2)
    filtered_points = {
        "x": las.x[filtered_mask],
        "y": las.y[filtered_mask],
        "z": las.z[filtered_mask],
        "classification": las.classification[filtered_mask],
    }
    
    print(f"File: {os.path.basename(las_path)}")
    print(f"Points count: {len(las.points)}")
    print(f"Unique labels: {unique_labels}")
    print(f"Counts per label: {label_counts}")
    print(f"Filtered points count: {len(filtered_points['x'])}")
    
    return filtered_points

# Process all LAS files and keep summaries
all_filtered_points = []
for las_file in las_files:
    filtered_points = process_las_file(las_file)
    all_filtered_points.append(filtered_points)

# Read GeoJSON file
geojson_data = gpd.read_file(geojson_path)
print("GeoJSON Info:")
print(geojson_data.head())

# Plot the first file for visualization
first_file_points = all_filtered_points[0]
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    first_file_points["x"][::100],
    first_file_points["y"][::100],
    first_file_points["z"][::100],
    c=first_file_points["classification"][::100],
    cmap="tab10",
    s=0.5
)
ax.set_title("Filtered Points Visualization (First File)")
plt.show()
