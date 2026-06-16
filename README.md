# House Price Predictor

## Problem Statement
Predict residential house sale prices from structural and location features.

## Dataset
- Source: Kaggle House Prices Competition
- Target: `SalePrice` (continuous, USD)

## Setup
```bash
# Clone the repository
git clone https://github.com/arapkorir1/house-price-predictorhttps://github.com/arapkorir1/house-price-predictor 
cd house-price-predictor

# Create the environment from the blueprint
conda env create -f environment.yml

# Activate the workspace
conda activate house-price
```

## Run Notebooks
```bash
jupyter lab
```

## Run API
```bash
uvicorn api.main:app --reload --port 8000
```

## Run Demo
```bash
streamlit run app/streamlit_app.py
```

