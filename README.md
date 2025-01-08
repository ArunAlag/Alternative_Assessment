# UNet++ and UNet Implementation

## Overview
This repository contains implementations of **UNet++** and **UNet** for image segmentation tasks using the COCO dataset. Users can either download the pre-trained models to inspect results or run the scripts to train and test the models themselves.

## File Structure
- `final_implementation.ipynb`: Jupyter Notebook for **UNet++**.
- `final_unet_implementation.ipynb`: Jupyter Notebook for **UNet**.
- `final_implementation.py`: Python script for **UNet++**.
- `final_unet_implementation.py`: Python script for **UNet**.
- `test/`, `train/`, `valid/`: Folders containing dataset files with **COCO annotations**.
- `unetplusplus_final_model.keras`: Pre-trained model for **UNet++**.
- `unet_small_best_model.keras`: Pre-trained model for **UNet**.
- `.gitattributes`: Configuration for Git LFS (to handle large files).

## How to Use
### 1️⃣ View Outputs Without Running the Code
To see the results directly:
1. Download the **Jupyter Notebooks** (`final_implementation.ipynb` and `final_unet_implementation.ipynb`).
2. Open them in **Jupyter Notebook** or **Google Colab**.

### 2️⃣ Run the Scripts Locally
If you want to train and test the models yourself:
1. Clone this repository:
   ```sh
   git clone https://github.com/ArunAlag/Alternative_Assessment.git
   cd Alternative_Assessment
   ```
2. Download the dataset folders (`test/`, `train/`, `valid/`).
3. Ensure that each dataset folder contains its respective **COCO annotation** file.
4. Run the script for the desired model:
   - **UNet++:**
     ```sh
     python final_implementation.py
     ```
   - **UNet:**
     ```sh
     python final_unet_implementation.py
     ```

## Pre-trained Models
For quick testing, you can use the provided pre-trained models:
- **UNet++ Model:** `unetplusplus_final_model.keras`
- **UNet Model:** `unet_small_best_model.keras`

## Environment Setup
This project was implemented in a **Python 3.10.7 virtual environment (`.venv`)**. Install the required dependencies using:
```sh
pip install tensorflow numpy matplotlib pycocotools
```

### Required Packages:
```python
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import Sequence
from tensorflow.keras import layers, models
from pycocotools.coco import COCO
import random
```

## Contributions & Feedback
Feel free to contribute or report any issues! 🚀

