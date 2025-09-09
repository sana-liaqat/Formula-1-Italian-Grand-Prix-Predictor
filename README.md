# Formula 1 Italian Grand Prix Predictor

Welcome to F1 Predictions 2025 – a data-driven approach to Formula 1!

This project combines the FastF1 API, historical race data, and a machine learning pipeline to simulate and predict race outcomes for the 2025 Formula 1 season. Using past performance, qualifying results, and engineered features, the model estimates finishing times and ranks drivers for each Grand Prix.

Whether you’re an F1 fan curious about data science or an ML enthusiast curious about F1, this project shows how machine learning can bring racing strategy and analytics to life.

## Project Overview

This repository contains a Gradient Boosting Machine Learning model that predicts race results based on:

- Past performance (historical race data)

- Qualifying times (2025 session data)

- Team/driver features engineered from F1 datasets

The model leverages:

- FastF1 API for historical race and qualifying data

- 2024 Italian GP race results for training

- 2025 qualifying session data for predictions

- Feature engineering techniques to improve accuracy

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

## Expected output
```bash
🏁 Predicted 2025 Italian GP Results 🏁
  Driver          Team  PredictedRaceTime (s)  PredictedPosition
3    NOR       McLaren              75.558324                  1
0    VER      Red Bull              75.894366                  2
1    LEC       Ferrari              75.914995                  3
4    PIA       McLaren              75.918563                  4
2    SAI       Ferrari              75.946225                  5
6    HAM      Mercedes              76.166732                  6
5    RUS      Mercedes              76.174746                  7
9    GAS        Alpine              76.706700                  8
8    ALB      Williams              76.717305                  9
7    ALO  Aston Martin              76.776086                 10
...
Model Performance:
MAE: 0.213 seconds
R² Score: 0.794
``` 

## Model Performance
The Mean Absolute Error (MAE) is used to evaluate how well the model predicts race times. Lower MAE values indicate more accurate predictions.
