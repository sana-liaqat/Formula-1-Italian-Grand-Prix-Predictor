import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from modeling import (
    evaluate_model, compare_models, plot_residuals, 
    plot_learning_curve, cross_validate_model, permutation_feature_importance
)
from data_collection import load_historical_data, get_weather_data
from config.config import *

def run_comprehensive_evaluation():
    """
    Run a comprehensive evaluation of the F1 prediction model
    """
    # Load and prepare data
    historical_data = load_historical_data(HISTORICAL_YEAR, HISTORICAL_RACE, "R")
    
    # Define team performance scores
    team_performance = {
        "Red Bull": 0.95,
        "Ferrari": 0.90,
        "McLaren": 0.87,
        "Mercedes": 0.85,
        "Aston Martin": 0.78,
        "Alpine": 0.72,
        "Williams": 0.70,
        "RB": 0.65,
        "Kick Sauber": 0.62,
        "Haas": 0.60
    }
    
    # Simulated qualifying data for 2025 Italian GP
    qualifying_data = pd.DataFrame({
        "Driver": ["VER", "LEC", "SAI", "NOR", "PIA", "RUS", "HAM", "ALO", "ALB", "GAS"],
        "Team": ["Red Bull", "Ferrari", "Ferrari", "McLaren", "McLaren", 
                "Mercedes", "Mercedes", "Aston Martin", "Williams", "Alpine"],
        "QualifyingTime (s)": [76.2, 76.5, 76.7, 76.8, 77.0, 77.1, 77.3, 77.5, 77.6, 77.8]
    })
    
    # Get weather data
    weather_data = get_weather_data(MONZA_LAT, MONZA_LON, "2025-09-07", OPENWEATHER_API_KEY)
    
    # Create features for prediction
    from modeling import create_features
    features = create_features(qualifying_data, weather_data, team_performance, historical_data)
    
    # For training, we'll use historical lap times as target
    target = historical_data["LapTime (s)"]
    
    # Prepare feature matrix and target vector
    X = features.drop(["Driver", "Team", "QualifyingTime (s)"], axis=1, errors="ignore")
    y = target
    
    # Handle missing values
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)
    
    # Scale features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    # Compare multiple models
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.svm import SVR
    
    models_to_test = {
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.1, max_depth=4, random_state=42
        ),
        'RandomForest': RandomForestRegressor(
            n_estimators=100, max_depth=5, random_state=42
        ),
        'LinearRegression': LinearRegression(),
        'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1)
    }
    
    results, best_model_name = compare_models(X_train, X_test, y_train, y_test, models_to_test)
    
    # Get the best model
    best_model = results[best_model_name]['model']
    y_pred = results[best_model_name]['y_pred']
    
    # Plot residuals for the best model
    plot_residuals(y_test, y_pred, best_model_name)
    
    # Plot learning curve for the best model
    plot_learning_curve(best_model, X_scaled, y, best_model_name)
    
    # Cross-validation for the best model
    mae_scores, r2_scores = cross_validate_model(best_model, X_scaled, y)
    
    # Permutation feature importance
    feature_names = X.columns.tolist()
    importance_df = permutation_feature_importance(best_model, X_test, y_test, feature_names)
    
    # Print final evaluation summary
    print("\n" + "="*60)
    print("FINAL EVALUATION SUMMARY")
    print("="*60)
    print(f"Best Model: {best_model_name}")
    print(f"Test MAE: {results[best_model_name]['mae']:.4f} seconds")
    print(f"Test R²: {results[best_model_name]['r2']:.4f}")
    print(f"CV MAE: {mae_scores.mean():.4f} (+/- {mae_scores.std() * 2:.4f})")
    print(f"CV R²: {r2_scores.mean():.4f} (+/- {r2_scores.std() * 2:.4f})")
    
    # Save the best model
    from modeling import save_model
    save_model(best_model, f"models/best_{best_model_name.lower()}_model.pkl")
    
    return best_model, results

if __name__ == "__main__":
    best_model, results = run_comprehensive_evaluation()
