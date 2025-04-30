import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


class ForestFireMLModel:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,  # Number of trees
            max_depth=10,  # Maximum tree depth
            min_samples_split=5,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_names = [
            'temperature', 'humidity', 'wind_speed',
            'rainfall', 'days_without_rain'
        ]

    def prepare_data(self, csv_path='fire_data.csv'):
        """
        Prepare training data from CSV
        """
        # Read the CSV file
        df = pd.read_csv(csv_path)

        # Ensure we have enough data
        if len(df) < 10:
            raise ValueError("Not enough training data. Need at least 10 records.")

        # Extract features and target
        X = df[self.feature_names]
        y = df['risk_level']

        return X, y

    def train(self, csv_path='fire_data.csv'):
        """
        Train the model on historical data
        """
        X, y = self.prepare_data(csv_path)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.model.fit(X_train_scaled, y_train)

        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)

        return {
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }

    def predict(self, features):
        """
        Predict fire risk based on input features
        """
        # Validate input features
        for feature in self.feature_names:
            if feature not in features:
                raise ValueError(f"Missing feature: {feature}")

        # Prepare input
        X = np.array([
            [features[feature] for feature in self.feature_names]
        ])

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Predict
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]

        # Get feature importances
        importances = self.model.feature_importances_
        feature_importance = dict(zip(
            self.feature_names,
            importances
        ))

        return {
            'risk_level': prediction,
            'probabilities': {
                'Low': probabilities[0],
                'Moderate': probabilities[1],
                'High': probabilities[2],
                'Extreme': probabilities[3]
            },
            'feature_importance': feature_importance
        }

    def analyze_feature_importance(self):
        """
        Detailed feature importance analysis
        """
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]

        return [
            {
                'feature': self.feature_names[i],
                'importance': importances[i]
            }
            for i in indices
        ]