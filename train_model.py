# train_model.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
import os
from datetime import datetime

# Parameters for synthetic data
N_SAMPLES = 1000
OUTPUT_FILE = 'training_data_synthetic.csv'
MODEL_FILE = 'forest_fire_model_latest.joblib'


def generate_synthetic_data(n_samples=1000):
    """Generate synthetic training data"""
    # Generate random environmental features
    temperature = np.random.uniform(10, 40, n_samples)  # 10-40°C
    humidity = np.random.uniform(20, 90, n_samples)  # 20-90%
    wind_speed = np.random.uniform(0, 50, n_samples)  # 0-50 km/h
    rainfall = np.random.exponential(scale=5, size=n_samples)  # Exponential distribution for rainfall
    days_without_rain = np.random.randint(0, 30, n_samples)  # 0-30 days

    # Calculate risk scores based on the same rules used in the application
    risk_scores = np.zeros(n_samples)

    # Temperature risk
    risk_scores += np.where(temperature >= 35, 4,
                            np.where(temperature >= 30, 3,
                                     np.where(temperature >= 25, 2,
                                              np.where(temperature >= 20, 1, 0))))

    # Humidity risk (lower humidity = higher risk)
    risk_scores += np.where(humidity < 30, 4,
                            np.where(humidity < 40, 3,
                                     np.where(humidity < 50, 2,
                                              np.where(humidity < 60, 1, 0))))

    # Wind risk
    risk_scores += np.where(wind_speed >= 30, 4,
                            np.where(wind_speed >= 20, 3,
                                     np.where(wind_speed >= 10, 2,
                                              np.where(wind_speed >= 5, 1, 0))))

    # Recent rainfall risk
    risk_scores += np.where(rainfall == 0, 2,
                            np.where(rainfall < 5, 1, 0))

    # Drought risk
    risk_scores += np.where(days_without_rain >= 14, 4,
                            np.where(days_without_rain >= 7, 3,
                                     np.where(days_without_rain >= 3, 2,
                                              np.where(days_without_rain >= 1, 1, 0))))

    # Determine risk levels
    risk_levels = np.where(risk_scores >= 15, 'Extreme',
                           np.where(risk_scores >= 10, 'High',
                                    np.where(risk_scores >= 5, 'Moderate', 'Low')))

    # Create dataframe
    df = pd.DataFrame({
        'temperature': temperature,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'rainfall': rainfall,
        'days_without_rain': days_without_rain,
        'risk_score': risk_scores,
        'risk_level': risk_levels
    })

    return df


def train_random_forest(df):
    """Train Random Forest model on synthetic data"""
    # Prepare features and target
    X = df[['temperature', 'humidity', 'wind_speed', 'rainfall', 'days_without_rain']]
    y = df['risk_level']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    joblib.dump(model, MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")

    # Also save the scaler for future use
    joblib.dump(scaler, 'scaler_latest.joblib')
    print("Scaler saved to scaler_latest.joblib")

    return model, scaler


def main():
    print("Generating synthetic training data...")
    df = generate_synthetic_data(N_SAMPLES)

    # Save to CSV
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {N_SAMPLES} synthetic data points to {OUTPUT_FILE}")

    # Train and save model
    print("Training Random Forest model...")
    model, scaler = train_random_forest(df)

    print("Done!")


if __name__ == "__main__":
    main()