
# LIDAR Point Transformer Pipeline

This repository implements a pipeline for tree detection and classification using LiDAR data, combining Local Maxima Filtering (LMF) and a Point Transformer neural network.

## Dataset

The dataset is freely available on Kaggle: [Tree Detection Lidar RGB](https://www.kaggle.com/datasets/sentinel3734/tree-detection-lidar-rgb/data). The data includes LiDAR scans and field survey information required for training and evaluation.

## Repository Structure

```plaintext
LIDAR-POINT-TRANSFORMER/
│
├── data/
│   ├── als/                # Raw ALS LiDAR data (downloaded from Kaggle)
│   ├── als_preprocessed/   # Directory created during preprocessing to store processed data
│   ├── ortho/              # Orthophoto imagery (optional for visualization)
│   └── field_survey.geojson # GeoJSON containing field survey data (ground truth)
│
├── notebooks/              # Jupyter notebooks for exploratory data analysis
├── results_lmf/            # Results from Local Maxima Filtering
├── results_pre/            # Preprocessing outputs and stats (generated by main_pre.py)
├── results_pyg/            # Results from the Point Transformer
│
├── src/                    # Source code modules
│   ├── __init__.py
│   ├── lmf_utils.py         # Utilities for Local Maxima Filtering
│   ├── pre_visualization.py # Visualization utilities for preprocessing
│   ├── pyg_data_preparation.py # Data preparation for PyG
│   ├── pyg_model.py         # PyG Point Transformer model definition
│   ├── pyg_train.py         # Training and testing pipeline
│   └── pyg_utils.py         # General utilities for PyG
│
├── main.py                 # Main script to run the entire pipeline
├── main_lmf.py             # Script for running the Local Maxima Filtering pipeline
├── main_pre.py             # Script for preprocessing ALS data and visualizations
├── main_pyg.py             # Script for the Point Transformer pipeline
│
├── README.md               # Project README
├── requirements.txt        # Python dependencies
└── Final Project.pdf       # Project definition document (not part of the repository)
```

## Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/lidar-point-transformer.git
cd lidar-point-transformer
```

### 2. Download the Dataset

Download the dataset from Kaggle [here](https://www.kaggle.com/datasets/sentinel3734/tree-detection-lidar-rgb/data).

Unzip the dataset into the `data/als` directory. Ensure the structure matches the following:

```plaintext
data/
├── als/                # Raw ALS files (from Kaggle)
├── ortho/              # Orthophoto imagery (optional)
└── field_survey.geojson # Field survey data for ground truth
```

You **do not need to create `als_preprocessed/`**, as it will be generated during preprocessing.

### 3. Install Dependencies

Set up your Python environment and install required packages:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Run the Entire Pipeline

Run the `main.py` script to execute all steps of the pipeline sequentially, including preprocessing, Local Maxima Filtering (LMF), and Point Transformer training and evaluation:

```bash
python main.py
```

This will:

1. Preprocess ALS data (`main_pre.py`) and create `als_preprocessed/`. It also generates visualizations of preprocessed data, saved in `results_pre/`.
2. Perform Local Maxima Filtering (`main_lmf.py`) and save results in `results_lmf/`.
3. Train and evaluate the Point Transformer model (`main_pyg.py`), saving results in `results_pyg/`.

### 5. Results

- **Preprocessing Outputs**: Available in the `results_pre/` directory, including visualizations of the preprocessed data and statistical summaries.
- **LMF Results**: Available in the `results_lmf/` directory, including detected trees and visualizations.
- **Point Transformer Results**: Available in the `results_pyg/` directory, including per-file metrics, plots, and GeoJSON exports.

---

### Notes

- Running `main.py` executes the entire pipeline. Individual steps can also be run separately using `main_pre.py`, `main_lmf.py`, or `main_pyg.py`.
- `Final Project.pdf` contains the project definition and objectives and was shared for evaluation purposes. It is not part of the repository's pipeline.
- The results generated from running the pipeline can be used for further visualization, analysis, or reporting.
