# Forest Fire Risk Prediction System

This project is a web-based system that predicts forest fire risk using machine learning. It integrates real-time weather data, satellite imagery, and advanced input features to estimate fire risk levels. The system includes support for distributed batch inference using Ray, simulating a parallel processing environment.

## Features

- Real-time weather data retrieval using the OpenWeatherMap API
- Satellite imagery access via NASA Earth API
- Machine learning model predictions of fire risk levels
- Feature importance visualization to interpret model behavior
- Parallelized batch prediction using Ray (local mode)
- Recent prediction history displayed on the frontend
- Interactive, responsive user interface with multiple input options

## Technology Stack

- **Frontend:** HTML, CSS, JavaScript (Vanilla)
- **Backend:** Flask (Python)
- **Machine Learning:** scikit-learn, joblib
- **Distributed Computing:** Ray (local mode)
- **Data Sources:** OpenWeatherMap, NASA Earth Imagery (Landsat)

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/forest-fire-predictor.git
    cd forest-fire-predictor
    ```

2. (Optional) Create a virtual environment:

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Create a `.env` file in the root directory and add your API keys:

    ```dotenv
    WEATHER_API_KEY=your_openweathermap_api_key
    NASA_API_KEY=your_nasa_api_key
    ```

## Usage

Start the Flask server:

```bash
python application.py
```

By default, the server will run on `http://localhost:8000`.

Access the UI through your browser at `http://localhost:8000`.

## API Endpoints

### `POST /api/predict`

Performs risk prediction using the trained model. Accepts both single and batch inputs.

**Request (JSON array):**

```json
[
  {
    "temperature": 30,
    "humidity": 40,
    "windSpeed": 20,
    "rainfall": 0,
    "pressure": 1010,
    "vegetation_type_encoded": 2,
    "elevation": 300,
    "firms_confidence": 70,
    "firms_brightness": 320
  }
]
```

**Response:**

```json
{
  "status": "success",
  "timestamp": "2025-04-30T13:05:12.942357",
  "batch_predictions": [
    {
      "risk_level": "Moderate",
      "probabilities": {
        "Low": 0.35,
        "Moderate": 0.46,
        "High": 0.18
      }
    }
  ]
}
```

### `POST /api/train-model`

Trains the model using the most recent `training_data_*.csv` file. Saves the model, scaler, and feature importance data.

### `POST /api/collect-data`

Triggers synthetic data collection from multiple geographic locations, saves it as a timestamped CSV.

### `GET /api/feature-importance`

Returns the most recently saved feature importances for display on the frontend.

### `GET /api/history`

Returns the 10 most recent predictions stored in `fire_data.csv`.

## Project Structure

```
forest-fire-predictor/
│
├── application.py              # Main Flask backend
├── distributed_predictor.py    # Ray-based distributed prediction logic
├── templates/index.html        # Frontend UI
├── fire_data.csv               # Stored prediction history
├── feature_importance.json     # Stored feature importances
├── forest_fire_model_*.joblib  # Trained model files
├── scaler_*.joblib             # Trained scaler files
├── .env                        # API keys (not committed)
└── requirements.txt
```

## Notes on Distributed Computing

Although this system uses Ray in local mode, the `predict_batch` method in `distributed_predictor.py` demonstrates how model inference can be parallelized across tasks. Each prediction task is executed via `ray.remote` and collected using `ray.get()`. This structure is ready for extension to true distributed clusters.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Author

Anindhya Kushagra
GitHub: [github.com/anindhya1](https://github.com/anindhya1)
