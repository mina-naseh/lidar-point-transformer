import logging
from src.pyg_data_preparation import prepare_data_with_transform
from src.pyg_train import train_model, test_model_per_file
import torch
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

npy_dir = "./data/als_preprocessed"
geojson_path = "./data/field_survey.geojson"
model_path = "./results_pyg/trained_point_transformer.pth"

def main():

    # Step 1: Prepare Data with k-NN Graph Transform
    print("Preparing data with k-NN graph transform...")  
    pyg_data_list = prepare_data_with_transform(npy_dir, geojson_path, k=16, radius=1.0)

    print(f"Data summary:")
    for idx, data in enumerate(pyg_data_list):
        print(f"Data {idx + 1}: x shape: {data.x.shape}, pos shape: {data.pos.shape}, "
              f"edge_index shape: {data.edge_index.shape}, y shape: {data.y.shape}")

    # Step 2: Split Data into Train and Test Sets
    split_idx = len(pyg_data_list) // 2
    train_data = pyg_data_list[:split_idx]
    test_data = pyg_data_list[split_idx:]

    logger.info(f"Split {len(pyg_data_list)} datasets into {len(train_data)} training and {len(test_data)} testing datasets.")

    # Step 3: Train the Model
    logger.info("Training the model...")
    model, test_loader, device = train_model(
        train_data=train_data,
        test_data=test_data,
        in_channels=3,  # Features are 3D coordinates
        out_channels=1,  # Binary classification (tree/non-tree)
        epochs=3,
        batch_size=1,
        lr=0.001
    )

    # Step 4: Save the Trained Model
    logger.info(f"Saving the trained model to {model_path}...")
    torch.save(model.state_dict(), model_path)


    # Test the model and calculate per-file metrics
    file_metrics = test_model_per_file(model, test_loader, device)

    with open("./results_pyg/test_metrics_per_file.json", "w") as f:
        json.dump(file_metrics, f, indent=4)
    print("Per-file test metrics saved to test_metrics_per_file.json")


    logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
