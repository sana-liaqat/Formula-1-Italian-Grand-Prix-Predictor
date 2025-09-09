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
