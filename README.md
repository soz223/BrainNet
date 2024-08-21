
# BrainNet

## Setup

To run the project, you first need to set up the environment. You can do this using either `pip` or `conda` by following the steps below:

### Using Conda

1. **Create the Conda environment** from the `environment.yml` file:
   ```bash
   conda env create -f environment.yml
   ```

2. **Activate the Conda environment**:
   ```bash
   conda activate your_env_name
   ```
   Replace `your_env_name` with the name specified in your `environment.yml` file, or simply use the default environment name.

### Using Pip

1. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv env
   ```

2. **Activate the virtual environment**:
   - On macOS/Linux:
     ```bash
     source env/bin/activate
     ```
   - On Windows:
     ```bash
     .\env\Scripts\activate
     ```

3. **Install the required packages** from the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

After setting up the environment, you can run the project using the following files:

- **Main Script**: The primary executable file is located at:
  ```
  classification/NeuroGraph/GNNs_ADNI.ipynb
  ```

- **Utilities**: The utility functions required by the main script are located at:
  ```
  classification/NeuroGraph/utils.py
  ```

Open the Jupyter Notebook `GNNs_ADNI.ipynb` to start running the analysis.

## Data

The edge data of ADNI is located in `data/ADNI/fmri_edge`. On GitHub, only cosine and pearson correlation data are provided. For more data, refer to the following link:
[Google Drive - Additional Data](https://drive.google.com/drive/folders/1ED1b7RoSdeqKnxfGUKM2zHUfqgP3fTzV)

