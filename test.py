import numpy as np
from src.point_cloud_utils import load_las_file



# points = load_las_file("./data/als/plot_01.las")
# print(points[:10])  # Inspect the first 10 points (X, Y, Z, classification)
# unique_classes = np.unique(points[:, -1])  # Check unique classification codes
# print(f"Unique classifications in plot_01: {unique_classes}")

import laspy

# Path to your LAS file
las_file_path = "./data/als/plot_01.las"

# Load the LAS file
las = laspy.read(las_file_path)

# Inspect the LAS header
print("LAS Header:")
print(las.header)

# List available point fields
print("\nAvailable Point Fields:")
print(las.point_format)

# Check if the classification field is present
if 'classification' in las.point_format.dimension_names:
    print("\nClassification data is available.")
    # Print the first 10 classifications for inspection
    print("First 10 Classification Values:", las.classification[:10])
else:
    print("\nClassification data is NOT available in this file.")




import os
import laspy
import numpy as np

# Directory containing LAS files
las_dir = "./data/als"

# Collect all LAS file paths
las_file_paths = sorted([os.path.join(las_dir, f) for f in os.listdir(las_dir) if f.endswith(".las")])

# Iterate over all LAS files and extract unique classifications
for las_file_path in las_file_paths:
    print(f"Processing file: {las_file_path}")
    las = laspy.read(las_file_path)
    classification_values = np.unique(las.classification)
    print(f"Unique Classification Values in {las_file_path}: {classification_values}")
