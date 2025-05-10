# distributed_predictor.py
import numpy as np
import joblib
import ray

# Initialize Ray
if not ray.is_initialized():
    ray.init(local_mode=True, ignore_reinit_error=True)

@ray.remote
def predict_single(model, scaler, feature_array, feature_names):
    import pandas as pd
    import os
    print(f"[Ray Task] PID: {os.getpid()} for features: {feature_array}")

    feature_df = pd.DataFrame([feature_array], columns=feature_names)
    feature_array_scaled = scaler.transform(feature_df)
    prediction = model.predict(feature_array_scaled)[0]
    probabilities = model.predict_proba(feature_array_scaled)[0]
    return prediction, probabilities

class DistributedPredictor:
    def __init__(self, model_path='forest_fire_model_latest.joblib', scaler_path='scaler_latest.joblib'):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.labels = self.model.classes_
        # Only the correct trained features:
        self.feature_names = [
            'temperature', 'humidity', 'wind_speed', 'rainfall',
            'pressure', 'vegetation_type_encoded',
            'elevation', 'firms_confidence', 'firms_brightness'
        ]

    def predict_batch(self, features_batch):
        futures = []
        for features in features_batch:
            feature_array = np.array([
                features['temperature'],
                features['humidity'],
                features['wind_speed'],
                features['rainfall'],
                features['pressure'],
                features['vegetation_type_encoded'],
                features['elevation'],
                features['firms_confidence'],
                features['firms_brightness']
            ])
            future = predict_single.remote(self.model, self.scaler, feature_array, self.feature_names)
            futures.append(future)

        results = ray.get(futures)
        predictions = []
        for pred_label, prob_array in results:
            pred_result = {
                'risk_level': pred_label,
                'probabilities': dict(zip(self.labels, prob_array))
            }
            predictions.append(pred_result)

        return predictions
