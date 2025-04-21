# Iris Flower Classification Project

This project implements a machine learning model to classify Iris flowers into three species based on their measurements. The model uses a Random Forest Classifier and includes additional features for data visualization and prediction.

## Features

- Data preprocessing and feature scaling
- Random Forest Classifier for species prediction
- Model evaluation with accuracy metrics and confusion matrix
- Data visualization tools
- Interactive prediction interface
- Model persistence for future use

## Requirements

Install the required packages using:
```bash
pip install -r requirements.txt
```

## Dataset

The project uses the Iris Flower Dataset from Kaggle. The dataset should be placed in the project directory as `IRIS.csv`.

## Usage

1. **Data Visualization**
   ```bash
   python visualize.py
   ```
   This will generate three visualization files:
   - pair_plot.png: Shows relationships between all features
   - box_plots.png: Displays distribution of each feature by species
   - correlation_heatmap.png: Shows correlations between features

2. **Model Training**
   ```bash
   python train.py
   ```
   This will:
   - Train the Random Forest Classifier
   - Evaluate the model
   - Save the trained model and scaler in the 'models' directory
   - Generate a confusion matrix plot

3. **Making Predictions**
   ```bash
   python predict.py
   ```
   This will start an interactive session where you can input flower measurements to get species predictions.

## Project Structure

- `train.py`: Main training script
- `predict.py`: Prediction interface
- `visualize.py`: Data visualization tools
- `requirements.txt`: Project dependencies
- `models/`: Directory containing saved models
- Generated visualization files

## Model Performance

The Random Forest Classifier typically achieves high accuracy (>95%) on the Iris dataset. The model's performance metrics and confusion matrix are displayed during training.

## Additional Features

- Feature scaling for better model performance
- Probability estimates for predictions
- Comprehensive data visualization
- Model persistence for future use
- Error handling and input validation