# distributed_model_trainer.py (UPDATED)
import pandas as pd
import numpy as np
import ray
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report

# Initialize Ray
ray.init(ignore_reinit_error=True)


@ray.remote
def train_single_fold(X_train, y_train, X_val, y_val):
    """
    Train and evaluate a Random Forest on a single fold
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_val_scaled)
    report = classification_report(y_val, y_pred, output_dict=True)

    return model, scaler, report


class DistributedModelTrainer:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits
        self.feature_names = [
            'temperature', 'humidity', 'wind_speed',
            'rainfall',
            'pressure', 'vegetation_type_encoded',
            'elevation', 'firms_confidence', 'firms_brightness'
        ]
        self.label_encoder = LabelEncoder()

    def load_and_prepare_data(self, csv_path):
        """
        Load CSV, encode categorical variables, create risk_level if missing,
        split features and target
        """
        df = pd.read_csv(csv_path)

        # Fill missing columns if needed
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0

        # Encode vegetation_type to numeric if it's not encoded yet
        if 'vegetation_type' in df.columns and 'vegetation_type_encoded' not in df.columns:
            df['vegetation_type_encoded'] = self.label_encoder.fit_transform(df['vegetation_type'])

        # --- 💥 Create risk_level if not present ---
        if 'risk_level' not in df.columns:
            print("⚡ Risk level not found, generating based on features...")

            risk_scores = np.zeros(len(df))

            # Temperature risk
            risk_scores += np.where(df['temperature'] >= 35, 4,
                                    np.where(df['temperature'] >= 30, 3,
                                             np.where(df['temperature'] >= 25, 2,
                                                      np.where(df['temperature'] >= 20, 1, 0))))

            # Humidity risk (lower humidity = higher risk)
            risk_scores += np.where(df['humidity'] < 30, 4,
                                    np.where(df['humidity'] < 40, 3,
                                             np.where(df['humidity'] < 50, 2,
                                                      np.where(df['humidity'] < 60, 1, 0))))

            # Wind speed risk
            risk_scores += np.where(df['wind_speed'] >= 30, 4,
                                    np.where(df['wind_speed'] >= 20, 3,
                                             np.where(df['wind_speed'] >= 10, 2,
                                                      np.where(df['wind_speed'] >= 5, 1, 0))))

            # Pressure risk (lower pressure = stormy weather, higher fire risk sometimes)
            risk_scores += np.where(df['pressure'] < 1000, 3,
                                    np.where(df['pressure'] < 1010, 2, 0))

            # Elevation risk (optional)
            risk_scores += np.where(df['elevation'] > 1000, 2, 0)

            # Dummy drought/fire factors since rainfall is missing
            risk_scores += np.where(df['firms_confidence'] > 30, 3, 0)

            # Assign risk level labels
            df['risk_level'] = np.where(risk_scores >= 15, 'Extreme',
                                        np.where(risk_scores >= 10, 'High',
                                                 np.where(risk_scores >= 5, 'Moderate', 'Low')))

        X = df[self.feature_names]
        y = df['risk_level']

        return X, y

    def cross_validate_and_train(self, csv_path):
        """
        Perform distributed cross-validation and final model training
        """
        X, y = self.load_and_prepare_data(csv_path)

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        futures = []

        for train_index, val_index in skf.split(X, y):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            # Start remote training
            future = train_single_fold.remote(X_train, y_train, X_val, y_val)
            futures.append(future)

        results = ray.get(futures)

        all_reports = [res[2] for res in results]

        # Now retrain on full data
        scaler_full = StandardScaler()
        X_full_scaled = scaler_full.fit_transform(X)
        model_full = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            n_jobs=-1,
            random_state=42
        )
        model_full.fit(X_full_scaled, y)

        # Save model and scaler
        joblib.dump(model_full, 'forest_fire_model_latest.joblib')
        joblib.dump(scaler_full, 'scaler_latest.joblib')
        print("✅ Model and scaler saved!")

        return {
            'fold_reports': all_reports,
            'model': model_full,
            'scaler': scaler_full
        }

if __name__ == "__main__":
    trainer = DistributedModelTrainer()
    trainer.cross_validate_and_train('fire_training_data_random_20250428_165030.csv')



