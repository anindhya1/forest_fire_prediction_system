# application.py
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
import requests
from dotenv import load_dotenv
import traceback
import json

# Machine Learning and Data Collection
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from distributed_predictor import DistributedPredictor

distributed_predictor = DistributedPredictor()

# Load environment variables
load_dotenv()

application = Flask(__name__)


# Advanced Data Collector
class AdvancedDataCollector:
    def __init__(self):
        """
        Initialize data collection from multiple sources
        """
        self.apis = {
            'openweathermap': self._collect_openweathermap_data
        }

    def collect_comprehensive_data(self, locations):
        """
        Collect data for multiple locations

        :param locations: List of (lat, lon) tuples
        :return: Consolidated DataFrame
        """
        all_data = []

        for lat, lon in locations:
            location_data = self._collect_location_data(lat, lon)
            all_data.append(location_data)

        return pd.concat(all_data, ignore_index=True)

    def _collect_location_data(self, lat, lon):
        """
        Collect data for a specific location from multiple sources
        """
        location_data = {
            'latitude': lat,
            'longitude': lon,
            'timestamp': datetime.now()
        }

        # Collect from each API
        for api_name, collector in self.apis.items():
            try:
                api_data = collector(lat, lon)
                location_data.update(api_data)
            except Exception as e:
                print(f"Error collecting data from {api_name}: {e}")

        # Add synthetic fire risk data
        location_data.update(self._generate_synthetic_fire_risk(location_data))

        return pd.DataFrame([location_data])

    def _collect_openweathermap_data(self, lat, lon):
        """
        Collect weather data from OpenWeatherMap
        """
        api_key = os.getenv('WEATHER_API_KEY')
        base_url = "https://api.openweathermap.org/data/2.5/weather"

        params = {
            'lat': lat,
            'lon': lon,
            'appid': api_key,
            'units': 'metric'
        }

        response = requests.get(base_url, params=params)

        if response.status_code == 200:
            data = response.json()
            return {
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'wind_speed': data['wind']['speed'] * 3.6,  # Convert to km/h
                'rainfall': data.get('rain', {}).get('1h', 0),
                'pressure': data['main']['pressure']
            }
        return {}

    def _generate_synthetic_fire_risk(self, weather_data):
        """
        Generate synthetic fire risk based on weather conditions
        """
        # Calculate risk score based on collected data
        risk_score = 0

        # Temperature risk
        temp = weather_data.get('temperature', 0)
        if temp >= 35:
            risk_score += 4
        elif temp >= 30:
            risk_score += 3
        elif temp >= 25:
            risk_score += 2

        # Humidity risk
        humidity = weather_data.get('humidity', 100)
        if humidity < 30:
            risk_score += 4
        elif humidity < 40:
            risk_score += 3

        # Wind speed risk
        wind_speed = weather_data.get('wind_speed', 0)
        if wind_speed >= 30:
            risk_score += 3
        elif wind_speed >= 20:
            risk_score += 2

        # Rainfall risk
        rainfall = weather_data.get('rainfall', 0)
        if rainfall == 0:
            risk_score += 3
        elif rainfall < 5:
            risk_score += 2

        # Determine risk level
        if risk_score >= 10:
            risk_level = 'Extreme'
        elif risk_score >= 7:
            risk_level = 'High'
        elif risk_score >= 4:
            risk_level = 'Moderate'
        else:
            risk_level = 'Low'

        return {
            'risk_level': risk_level,
            'risk_score': risk_score
        }


# Distributed Model Trainer
class DistributedModelTrainer:
    def __init__(self, n_workers=4):
        self.n_workers = n_workers
        self.feature_names = [
            'temperature', 'humidity', 'wind_speed',
            'rainfall', 'risk_score'
        ]

    def load_data(self, csv_path):
        """
        Load and prepare training data
        """
        df = pd.read_csv(csv_path)

        # Ensure data is sufficient
        if len(df) < 100:
            raise ValueError("Insufficient training data")

        # Split features and target
        X = df[self.feature_names]
        y = df['risk_level']

        return X, y

    def train_model(self, csv_path):
        """
        Train a single model
        """
        X, y = self.load_data(csv_path)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            n_jobs=-1  # Use all available cores
        )
        model.fit(X_train_scaled, y_train)

        # Evaluate
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)

        return {
            'model': model,
            'scaler': scaler,
            'performance': classification_report(y_test, y_pred),
            'feature_importance': dict(zip(
                self.feature_names,
                model.feature_importances_
            ))
        }


# Initialize data collector and model trainer
data_collector = AdvancedDataCollector()
model_trainer = DistributedModelTrainer()


# Existing Weather and Satellite Imagery Functions
def get_weather_data(lat, lon):
    """
    Fetch real-time weather data from OpenWeatherMap API
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
                'wind_speed': data['wind']['speed'] * 3.6,
                'rainfall': data.get('rain', {}).get('1h', 0),
                'pressure': data['main']['pressure']
            }


            return weather_data
        else:
            return {"error": f"API returned status code {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def get_satellite_imagery(lat, lon, date=None):
    """
    Fetch satellite imagery for a specific location
    """
    # If no date provided, use current date minus 1 month
    if date is None:
        one_month_ago = datetime.now() - timedelta(days=30)
        date = one_month_ago.strftime("%Y-%m-%d")

    # Using NASA EOSDIS API
    api_key = os.getenv('NASA_API_KEY')
    if not api_key:
        return {
            "status": "error",
            "message": "NASA API key not configured. Please check environment variables."
        }
    base_url = "https://api.nasa.gov/planetary/earth/assets"

    params = {
        "lat": lat,
        "lon": lon,
        "date": date,
        "dim": 0.15,  # Width/height of image in degrees
        "api_key": api_key
    }

    try:
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            return {
                "status": "success",
                "image_url": data.get("url"),
                "date": data.get("date"),
                "cloud_score": data.get("cloud_score")
            }
        else:
            return {"status": "error", "message": f"API returned status code {response.status_code}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# Routes
@application.route('/')
def home():
    return render_template('index.html')


@application.route('/api/collect-data', methods=['POST'])
def collect_training_data():
    """
    Endpoint to trigger comprehensive data collection
    """
    try:
        # Define locations to collect data
        locations = [
            (40.7128, -74.0060),  # New York
            (34.0522, -118.2437),  # Los Angeles
            (51.5074, -0.1278),  # London
            (-33.8688, 151.2093),  # Sydney
            (37.7749, -122.4194)  # San Francisco
        ]

        # Collect data
        data = data_collector.collect_comprehensive_data(locations)

        # Save collected data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f'training_data_{timestamp}.csv'
        data.to_csv(output_file, index=False)

        return jsonify({
            'status': 'success',
            'records_collected': len(data),
            'file_saved': output_file
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@application.route('/api/feature-importance', methods=['GET'])
def get_feature_importance():
    """Serve feature importance to frontend"""
    try:
        with open("feature_importance.json", "r") as f:
            importance = json.load(f)
        return jsonify({'status': 'success', 'feature_importance': importance})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@application.route('/api/train-model', methods=['POST'])
def train_model():
    """
    Trigger model training
    """
    try:
        # Get training data path from request or use most recent
        data_files = [f for f in os.listdir('.') if f.startswith('training_data_') and f.endswith('.csv')]

        if not data_files:
            raise ValueError("No training data found")

        # Use the most recent training data file
        latest_data_file = max(data_files)

        # Train model
        training_result = model_trainer.train_model(latest_data_file)

        # Save feature importance for frontend
        with open("feature_importance.json", "w") as f:
            json.dump(training_result['feature_importance'], f)

        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f'forest_fire_model_{timestamp}.joblib'
        joblib.dump(training_result['model'], model_filename)

        return jsonify({
            'status': 'success',
            'model_saved': model_filename,
            'performance': training_result['performance'],
            'feature_importance': training_result['feature_importance']
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


def calculate_risk_score(data):
    """
    Simple synthetic risk score calculation.
    """
    risk_score = 0
    temperature = data.get('temperature', 0)
    humidity = data.get('humidity', 100)
    wind_speed = data.get('windSpeed', 0)
    rainfall = data.get('rainfall', 0)

    if temperature >= 35:
        risk_score += 4
    elif temperature >= 30:
        risk_score += 3
    elif temperature >= 25:
        risk_score += 2

    if humidity < 30:
        risk_score += 4
    elif humidity < 40:
        risk_score += 3

    if wind_speed >= 30:
        risk_score += 3
    elif wind_speed >= 20:
        risk_score += 2

    if rainfall == 0:
        risk_score += 3
    elif rainfall < 5:
        risk_score += 2

    return risk_score



@application.route('/api/predict', methods=['POST'])
def predict():
    """
    Distributed prediction endpoint — now supports batch input
    """
    try:
        data = request.get_json()
        print(f"📥 Incoming /api/predict data: {data}")

        # Handle batch input
        if isinstance(data, list):
            features_batch = []
            for item in data:
                features_batch.append({
                    'temperature': float(item.get('temperature', 0)),
                    'humidity': float(item.get('humidity', 0)),
                    'wind_speed': float(item.get('windSpeed', 0)),
                    'rainfall': float(item.get('rainfall', 0)),
                    'pressure': float(item.get('pressure', 1013)),
                    'vegetation_type_encoded': float(item.get('vegetation_type_encoded', 0)),
                    'elevation': float(item.get('elevation', 0)),
                    'firms_confidence': float(item.get('firms_confidence', 0)),
                    'firms_brightness': float(item.get('firms_brightness', 0))
                })

            prediction_results = distributed_predictor.predict_batch(features_batch)
            print(f"🔮 Batch prediction results: {prediction_results}")
            return jsonify({
                'status': 'success',
                'batch_predictions': prediction_results,
                'timestamp': datetime.now().isoformat()
            })

        # Handle single input (same as before)
        features = {
            'temperature': float(data.get('temperature', 0)),
            'humidity': float(data.get('humidity', 0)),
            'wind_speed': float(data.get('windSpeed', 0)),
            'rainfall': float(data.get('rainfall', 0)),
            'pressure': float(data.get('pressure', 1013)),
            'vegetation_type_encoded': float(data.get('vegetation_type_encoded', 0)),
            'elevation': float(data.get('elevation', 0)),
            'firms_confidence': float(data.get('firms_confidence', 0)),
            'firms_brightness': float(data.get('firms_brightness', 0))
        }

        prediction_results = distributed_predictor.predict_batch([features])
        prediction = prediction_results[0]

        row = {
            'timestamp': datetime.now().isoformat(),
            'temperature': features['temperature'],
            'humidity': features['humidity'],
            'wind_speed': features['wind_speed'],
            'rainfall': features['rainfall'],
            'risk_level': prediction['risk_level']
        }
        pd.DataFrame([row]).to_csv('fire_data.csv', mode='a', header=not os.path.exists('fire_data.csv'), index=False)

        return jsonify({
            'riskLevel': prediction['risk_level'],
            'riskProbabilities': prediction['probabilities'],
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        print("❌ Exception occurred in /api/predict:")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@application.route('/api/history', methods=['GET'])
def history():
    """Get prediction history"""
    try:
        # Define the CSV file path
        MOCK_DB_FILE = 'fire_data.csv'

        # Check if file exists
        if not os.path.exists(MOCK_DB_FILE):
            return jsonify([])

        # Read the CSV file
        df = pd.read_csv(MOCK_DB_FILE)
        records = df.tail(10).to_dict('records')
        return jsonify(records)
    except Exception as e:
        print(f"Error fetching history: {e}")
        return jsonify([])

def enrich_with_advanced_inputs(lat, lon):
    """
    Return simulated or default values for advanced inputs.
    """
    return {
        'vegetationIndex': round(np.random.uniform(0.3, 0.8), 2),
        'fireDensity': round(np.random.uniform(0, 5), 1),
        'elevation': 250,  # placeholder or use lat/lon-based API
        'firms_confidence': 60 + int(np.random.uniform(0, 40)),
        'firms_brightness': 300 + int(np.random.uniform(0, 50)),
        'vegetation_type_encoded': int(np.random.choice([0, 1, 2, 3]))
    }

@application.route('/api/weather', methods=['POST'])
def weather():
    """Fetch weather data for a location"""
    data = request.json
    lat = data.get('latitude')
    lon = data.get('longitude')

    if not lat or not lon:
        return jsonify({"error": "Latitude and longitude are required"}), 400

    weather_data = get_weather_data(lat, lon)
    enriched = enrich_with_advanced_inputs(lat, lon)
    weather_data.update(enriched)
    return jsonify(weather_data)


@application.route('/api/satellite', methods=['POST'])
def satellite():
    """Fetch satellite imagery for a location"""
    data = request.json
    lat = data.get('latitude')
    lon = data.get('longitude')
    date = data.get('date')  # Optional

    if not lat or not lon:
        return jsonify({"status": "error", "message": "Latitude and longitude are required"}), 400

    satellite_data = get_satellite_imagery(float(lat), float(lon), date)
    return jsonify(satellite_data)

if __name__ == '__main__':
    # Get port from environment variable or default to 8000
    port = int(os.environ.get('PORT', 8000))
    application.run(host='0.0.0.0', port=port)