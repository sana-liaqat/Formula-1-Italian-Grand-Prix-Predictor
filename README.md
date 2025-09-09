# Formula 1 Italian Grand Prix Predictor

A machine learning project that predicts Formula 1 race results for the Italian Grand Prix at Monza.

## Project Overview

This project uses historical F1 data, weather information, and team performance metrics to predict race outcomes for the Italian GP. The system employs a Gradient Boosting Regressor to forecast race times based on various features including qualifying performance, historical sector times, weather conditions, and team data.

## Features

- Historical data collection from FastF1 API
- Weather data integration from OpenWeatherMap
- Feature engineering for driver and team performance
- Machine learning model training and evaluation
- Visualization of predictions and feature importance

## Dependencies
- fastf1
- numpy
- pandas
- scikit-learn
- matplotlib

## Usage
Run the prediction script
```bash
jupyter notebook notebooks/2025_f1_italian_gp_prediction.ipynb
``` 

## Results
The model predicts race times for the 2025 Italian Grand Prix with visualizations showing:

- Predicted race results
- Qualifying vs race pace comparison
- Feature importance analysis
- Team performance comparisons

## Model Performance
The Mean Absolute Error (MAE) is used to evaluate how well the model predicts race times. Lower MAE values indicate more accurate predictions.
