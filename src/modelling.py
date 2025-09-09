import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

def create_features(qualifying_data, weather_data, team_performance, historical_data=None):
    """Create feature set for prediction"""
  
    features = qualifying_data.copy()
    
    # Add weather data
    features["Temperature"] = weather_data["temperature"]
    features["Humidity"] = weather_data["humidity"]
    features["RainProbability"] = weather_data["rain_probability"]
    features["IsRainy"] = 1 if weather_data["rain_probability"] > 0.5 else 0
    
    # Add team performance
    features["TeamPerformance"] = features["Team"].map(team_performance)
    
    # Merge with historical sector times
    if historical_data is not None:
        historical_avgs = historical_data.groupby("Driver").agg({
            "Sector1Time (s)": "mean",
            "Sector2Time (s)": "mean", 
            "Sector3Time (s)": "mean",
            "Consistency": "mean"
        }).reset_index()
        
        features = features.merge(historical_avgs, on="Driver", how="left")
    
    return features

def train_model(features, target, test_size=0.2, random_state=42, 
                n_estimators=150, learning_rate=0.1, max_depth=4):
    """Train the prediction model"""
    # Separate features and target
    X = features.drop(["Driver", "Team", "QualifyingTime (s)"], axis=1, errors="ignore")
    y = target
    
    # Handle missing values
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )
    
    # Train model
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"MAE: {mae:.3f} seconds")
    print(f"R² Score: {r2:.3f}")
    
    # Store feature importance
    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    return model, mae, r2, feature_importance

def predict_race(model, features):
    """Predict race results"""
    # Prepare features for prediction
    X = features.drop(["Driver", "Team", "QualifyingTime (s)"], axis=1, errors="ignore")
    
    # Handle missing values
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.transform(X_imputed)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    
    # Create results dataframe
    results = features[["Driver", "Team", "QualifyingTime (s)"]].copy()
    results["PredictedRaceTime (s)"] = predictions
    results = results.sort_values("PredictedRaceTime (s)")
    results["PredictedPosition"] = range(1, len(results) + 1)
    
    return results

def save_model(model, filepath):
    """Save the trained model to file"""
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")
    
def load_model(filepath):
    """Load a trained model from file"""
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model
