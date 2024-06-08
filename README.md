# Repository Overview

This repository contains various scripts and notebooks for data understanding, processing, and modeling, as well as for managing the MLflow and Jupyter container setups. Below is a detailed description of each component.

## Data Understanding

- **Exploratory Data Analysis (EDA):**
  - Refer to the `eda.ipynb` notebook for initial data exploration and understanding.

## Data Processing

- **Scripts:**
  - `data_processing.py`: Script for general data processing tasks.
  - `external_data.py`: Script for handling external data sources.

## Repository Structure

The repository is organized into three main folders, each serving a different purpose:

1. **Modeling:**
   - **Imputation and Encoding:**
     - `train_script.py`: Contains functions for data imputation and encoding, as well as the complete training pipeline.
   - **Clustering:**
     - `clustering.ipynb`: Notebook dedicated to clustering tasks.
     - `train_script.py`: Also used for clustering purposes.
   - **Bootstrapping:**
     - `bootstrap.py`: Script for bootstrapping.
     - `cluster_evaluation.py`: Contains bootstrapping logic for cluster evaluation and the assessment of clustering results.
   - **Baseline:**
     - `baseline_boot.py`: Contains the baseline with the bootstrap
2. **MLflow:**
   - This folder contains all files and configurations related to MLflow, which is used for tracking experiments and managing model deployments.

3. **Jupyter Container:**
   - Contains the setup and configuration files necessary for running the Jupyter notebook environment within a container.

## Parameter Management

- **JSON Files:**
  - These files store parameters for various algorithms. Parameters can be modified as needed for different runs.
  - **Running the Scripts:**
    - The `train_script.py` can be executed with different arguments to run various experiments. An example of how to run this script with different parameters is provided within the file.

## Pipeline Overview

- The `train_script.py` encapsulates the entire pipeline from data processing to model training and evaluation.
