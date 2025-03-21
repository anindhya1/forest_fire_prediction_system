# application.py
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import requests
from dotenv import load_dotenv

# Load environment variables (optional)
load_dotenv()

application = Flask(__name__)

# Mock database using CSV file
MOCK_DB_FILE = 'fire_data.csv'

# Initialize CSV if it doesn't exist
if not os.path.exists(MOCK_DB_FILE):
    data = {
        'timestamp': [],
        'temperature': [],
        'humidity': [],
        'wind_speed': [],
        'rainfall': [],
        'days_without_rain': [],
        'risk_level': []
    }
    pd.DataFrame(data).to_csv(MOCK_DB_FILE, index=False)


# Simple ML model
class ForestFireModel:
    def predict(self, features):
        """
        Simple prediction based on heuristics
        features: dict with temperature, humidity, wind_speed, rainfall, days_without_rain
        """
        risk_score = 0

        # Temperature risk
        if features['temperature'] >= 35:
            risk_score += 4
        elif features['temperature'] >= 30:
            risk_score += 3
        elif features['temperature'] >= 25:
            risk_score += 2
        elif features['temperature'] >= 20:
            risk_score += 1

        # Humidity risk (lower humidity = higher risk)
        if features['humidity'] < 30:
            risk_score += 4
        elif features['humidity'] < 40:
            risk_score += 3
        elif features['humidity'] < 50:
            risk_score += 2
        elif features['humidity'] < 60:
            risk_score += 1

        # Wind risk
        if features['wind_speed'] >= 30:
            risk_score += 4
        elif features['wind_speed'] >= 20:
            risk_score += 3
        elif features['wind_speed'] >= 10:
            risk_score += 2
        elif features['wind_speed'] >= 5:
            risk_score += 1

        # Recent rainfall risk
        if features['rainfall'] == 0:
            risk_score += 2
        elif features['rainfall'] < 5:
            risk_score += 1

        # Drought risk
        if features['days_without_rain'] >= 14:
            risk_score += 4
        elif features['days_without_rain'] >= 7:
            risk_score += 3
        elif features['days_without_rain'] >= 3:
            risk_score += 2
        elif features['days_without_rain'] >= 1:
            risk_score += 1

        # Calculate risk level
        if risk_score >= 15:
            return 'Extreme'
        elif risk_score >= 10:
            return 'High'
        elif risk_score >= 5:
            return 'Moderate'
        else:
            return 'Low'


# Initialize model
model = ForestFireModel()


# Optional: Weather API Integration
# Weather API functionality
def get_weather_data(lat, lon):
    """
    Fetch real-time weather data from OpenWeatherMap API

    Parameters:
    lat (float): Latitude
    lon (float): Longitude

    Returns:
    dict: Weather data including temperature, humidity, wind speed, and rainfall
    """
    api_key = os.getenv('WEATHER_API_KEY')
    if not api_key:
        return {"error": "Weather API key not configured"}

    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=metric&appid={api_key}"

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()

            # Extract relevant weather data
            weather_data = {
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'wind_speed': data['wind']['speed'] * 3.6,  # Convert m/s to km/h
                'rainfall': data.get('rain', {}).get('1h', 0)  # Rainfall in last hour, default 0
            }

            return weather_data
        else:
            return {"error": f"API returned status code {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

@application.route('/')
def home():
    return render_template('index.html')


@application.route('/api/predict', methods=['POST'])
def predict():
    data = request.json

    # Process input data
    features = {
        'temperature': float(data.get('temperature', 0)),
        'humidity': float(data.get('humidity', 0)),
        'wind_speed': float(data.get('windSpeed', 0)),
        'rainfall': float(data.get('rainfall', 0)),
        'days_without_rain': float(data.get('daysWithoutRain', 0))
    }

    # Make prediction
    risk_level = model.predict(features)

    # Save to "database"
    df = pd.read_csv(MOCK_DB_FILE)
    new_row = {
        'timestamp': datetime.now().isoformat(),
        'temperature': features['temperature'],
        'humidity': features['humidity'],
        'wind_speed': features['wind_speed'],
        'rainfall': features['rainfall'],
        'days_without_rain': features['days_without_rain'],
        'risk_level': risk_level
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(MOCK_DB_FILE, index=False)

    return jsonify({
        'riskLevel': risk_level,
        'timestamp': datetime.now().isoformat()
    })


@application.route('/api/history', methods=['GET'])
def history():
    df = pd.read_csv(MOCK_DB_FILE)
    records = df.tail(10).to_dict('records')
    return jsonify(records)


@application.route('/api/weather', methods=['POST'])
def weather():
    data = request.json
    lat = data.get('latitude')
    lon = data.get('longitude')

    if not lat or not lon:
        return jsonify({"error": "Latitude and longitude are required"}), 400

    weather_data = get_weather_data(float(lat), float(lon))
    return jsonify(weather_data)


if __name__ == '__main__':
    # Get port from environment variable or default to 8000
    port = int(os.environ.get('PORT', 8000))
    application.run(host='0.0.0.0', port=port)