# Tubes 2 AI

## Overview
This project implements three machine learning algorithms (KNN, Gaussian Naive-Bayes, and ID3) from scratch to classify network attacks using the UNSW-NB15 dataset.


## Features
1. KNN Implementation:
   - Customizable number of neighbors
   - Multiple distance metrics (Euclidean, Manhattan, Minkowski)

2. Gaussian Naive-Bayes Implementation:
   - Probability calculation for continuous features
   - Gaussian distribution assumption
   - Model save/load functionality (`.pkl` format)

3. ID3 Implementation:
   - Entropy-based decision tree
   - Handling of numerical attributes
   - Model save/load functionality (`.pkl` format)

4. Data Preprocessing:
   - Missing value imputation (median for numeric, constant for categorical)
   - Feature encoding with One-Hot Encoding
   - Outlier handling using mean replacement
   - Feature engineering including:
   - Data normalization using Min-Max scaling

## Setup and Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/raflyhangga/tubes2-ai
cd tubes2-ai
```

### Step 2: Set Up Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
# Run a simple Python script to verify installations
python -c "import numpy; import pandas; import sklearn; print('Setup successful!')"
```

### Step 5: Launch Jupyter Notebook
Install the Python extension in VS Code
Open the project folder in VS Code
Navigate to the src file
Select the ipynb file
Select the kernel will be used
Run all the file


## Model Comparison
The project includes custom implementations (`src/scratch`)

## Team Members and Contributions
1. Agil Fadillah Sabri (13522006)
   - ID3 implementation

2. Raden Rafly Hanggaraksa Budiarto (13522014)
   - Data cleaning and preprocessing

3. Bastian H Suryapratama (13522034)
   - KNN implementation

4. Moh Fairuz Alauddin Yahya (13522057)
   - Naive Bayes implementation
