import fastf1
import pandas as pd
import requests
from config.config import *

def load_historical_data(year, race_number, session_type="R"):
    """Load historical race data from FastF1"""
    print(f"Loading {year} race data...")
    session = fastf1.get_session(year, race_number, session_type)
    session.load()
    
    # Get lap data
    laps = session.laps.copy()
    laps = laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time", 
                 "Compound", "TyreLife", "FreshTyre", "Stint", "TrackStatus", "Position"]]
    
    # Only use green flag laps
    laps = laps[laps["TrackStatus"] == "1"]
    laps.dropna(subset=["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"], inplace=True)
    
    # Convert times to seconds
    for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        laps[f"{col} (s)"] = laps[col].dt.total_seconds()
    
    # Calculate consistency (standard deviation of lap times)
    consistency = laps.groupby("Driver")["LapTime (s)"].std().reset_index()
    consistency.columns = ["Driver", "Consistency"]
    
    # Get average lap times by driver
    avg_times = laps.groupby("Driver").agg({
        "LapTime (s)": "mean",
        "Sector1Time (s)": "mean",
        "Sector2Time (s)": "mean",
        "Sector3Time (s)": "mean",
        "TyreLife": "mean",
        "Stint": "max",
        "Position": "min"
    }).reset_index()
    
    # Merge with consistency data
    historical_data = pd.merge(avg_times, consistency, on="Driver")
    
    # Add team information
    driver_teams = {}
    for driver in laps["Driver"].unique():
        try:
            driver_teams[driver] = laps[laps["Driver"] == driver].iloc[0]["Team"]
        except:
            driver_teams[driver] = "Unknown"
            
    historical_data["Team"] = historical_data["Driver"].map(driver_teams)
    
    print(f"Loaded data for {len(historical_data)} drivers")
    return historical_data

def get_weather_data(lat, lon, race_date, api_key=None):
    """Get weather data from OpenWeatherMap API or use simulated data"""
    # If no API key, use simulated data for Monza
    if api_key is None:
        print("Using simulated weather data for Monza")
        return {
            "temperature": 26.5,
            "humidity": 65,
            "rain_probability": 0.1,
            "conditions": "Clear"
        }
    
    try:
        base_url = "http://api.openweathermap.org/data/2.5/forecast"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": api_key,
            "units": "metric"
        }
        
        response = requests.get(base_url, params=params)
        weather_data = response.json()
        
        # Find forecast closest to race time (typically 15:00 local)
        race_time = f"{race_date} 15:00:00"
        forecast_data = None
        
        for forecast in weather_data["list"]:
            if forecast["dt_txt"] == race_time:
                forecast_data = forecast
                break
                
        if forecast_data:
            weather_info = {
                "temperature": forecast_data["main"]["temp"],
                "humidity": forecast_data["main"]["humidity"],
                "rain_probability": forecast_data.get("pop", 0),
                "conditions": forecast_data["weather"][0]["main"]
            }
            return weather_info
    except:
        pass
    
    # Return default values if API call fails
    return {
        "temperature": 26.5,
        "humidity": 65,
        "rain_probability": 0.1,
        "conditions": "Clear"
    }
