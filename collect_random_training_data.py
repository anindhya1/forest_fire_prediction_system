# collect_random_training_data.py
import random
from advanced_data_collector import AdvancedDataCollector
from datetime import datetime
import pandas as pd
import time

def generate_random_locations(n_samples=100):
    """
    Generate random (latitude, longitude) tuples
    """
    locations = []
    for _ in range(n_samples):
        lat = random.uniform(-60, 60)  # Avoid polar regions (data unreliable)
        lon = random.uniform(-180, 180)
        locations.append((lat, lon))
    return locations

def main():
    collector = AdvancedDataCollector()

    # Generate 100 random locations
    random_locations = generate_random_locations(100)

    print(f"🌎 Collecting data for {len(random_locations)} random locations...")

    # Collect data
    collected_data = collector.collect_comprehensive_data(random_locations)

    # Save CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'fire_training_data_random_{timestamp}.csv'
    collected_data.to_csv(filename, index=False)

    print(f"✅ Data collection complete! Saved to {filename}")

if __name__ == "__main__":
    main()
