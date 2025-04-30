# data_manager.py
import os
import pandas as pd
from datetime import datetime


class DataManager:
    def __init__(self, base_dir='training_data'):
        """
        Manage training data collection and versioning
        """
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def save_collected_data(self, data):
        """
        Save collected data with timestamp
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'fire_training_data_{timestamp}.csv'
        filepath = os.path.join(self.base_dir, filename)

        data.to_csv(filepath, index=False)
        return filepath

    def list_training_datasets(self):
        """
        List available training datasets
        """
        return sorted([
            f for f in os.listdir(self.base_dir)
            if f.startswith('fire_training_data_') and f.endswith('.csv')
        ])

    def get_latest_dataset(self):
        """
        Get the most recent training dataset
        """
        datasets = self.list_training_datasets()
        return os.path.join(self.base_dir, datasets[-1]) if datasets else None