import logging
import os
import shutil
from src.pyg_data_preparation import prepare_data_with_transform
from src.pyg_train import train_model, test_model_per_file_with_visualization
import torch
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

npy_dir = "./data/als_preprocessed"
geojson_path = "./data/field_survey.geojson"
model_path = "./results_pyg/trained_point_transformer.pth"

def setup_results_directory(directory):
    """
    Ensures that the results directory exists and is empty.

    Args:
        directory (str): Path to the results directory.
    """
    if os.path.exists(directory):
        logger.info(f"Clearing existing directory: {directory}")
        shutil.rmtree(directory)
    os.makedirs(directory)
    logger.info(f"Created directory: {directory}")

def main():
    setup_results_directory("./results_pyg")

    pyg_data_list = prepare_data_with_transform(npy_dir, geojson_path, k=16, radius=1.0)

    print(f"Data summary:")
    for idx, data in enumerate(pyg_data_list):
        print(f"Data {idx + 1}: x shape: {data.x.shape}, pos shape: {data.pos.shape}, "
              f"edge_index shape: {data.edge_index.shape}, y shape: {data.y.shape}")

    split_idx = int(len(pyg_data_list) * 0.8)  # 80% for training
    train_data = pyg_data_list[:split_idx]
    test_data = pyg_data_list[split_idx:]  # Remaining 20% for testing

    logger.info(f"Split {len(pyg_data_list)} datasets into {len(train_data)} training and {len(test_data)} testing datasets.")

    model, test_loader, device = train_model(
        train_data=train_data,
        test_data=test_data,
        in_channels=3,
        out_channels=1,
        epochs=20,
        batch_size=1,
        lr=0.001
    )

    logger.info(f"Saving the trained model to {model_path}...")
    torch.save(model.state_dict(), model_path)

    file_metrics = test_model_per_file_with_visualization(model, test_loader, device, save_dir="./results_pyg")

    with open("./results_pyg/test_metrics_per_file.json", "w") as f:
        json.dump(file_metrics, f, indent=4)
    print("Per-file test metrics and visualizations saved to ./results_pyg")

    logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
