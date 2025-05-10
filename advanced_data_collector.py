# advanced_data_collector.py (Fixed for Free APIs)
import os
import pandas as pd
import requests
from datetime import datetime, timedelta
import time
from dotenv import load_dotenv
import zipfile
import io


# Load environment variables
load_dotenv()


class AdvancedDataCollector:
    def __init__(self):
        self.weather_api_key = os.getenv('WEATHER_API_KEY')
        self.earthdata_token = os.getenv('NASA_EARTHDATA_TOKEN')
        self.elevation_api = "https://api.open-elevation.com/api/v1/lookup"

    def collect_comprehensive_data(self, locations):
        all_data = []

        for lat, lon in locations:
            location_data = self._collect_location_data(lat, lon)
            all_data.append(location_data)

            time.sleep(1)  # Small delay to avoid rate limits

        return pd.concat(all_data, ignore_index=True)

    def _collect_location_data(self, lat, lon):
        location_data = {
            'latitude': lat,
            'longitude': lon,
            'timestamp': datetime.now()
        }

        try:
            location_data.update(self._collect_weather_data(lat, lon))
        except Exception as e:
            print(f"⚠️ Weather fetch failed: {e}")
            location_data.update({
                'temperature': None,
                'humidity': None,
                'wind_speed': None,
                'pressure': None
            })

        try:
            location_data.update(self._collect_elevation(lat, lon))
        except Exception as e:
            print(f"⚠️ Elevation fetch failed: {e}")
            location_data.update({'elevation': None})

        try:
            location_data.update(self._collect_firms_data(lat, lon))
        except Exception as e:
            print(f"⚠️ FIRMS fetch failed: {e}")
            location_data.update({
                'firms_confidence': None,
                'firms_brightness': None
            })

        # Dummy vegetation type (optional real later)
        location_data.update(self._collect_vegetation(lat, lon))

        return pd.DataFrame([location_data])

    def _collect_weather_data(self, lat, lon):
        base_url = "https://api.openweathermap.org/data/2.5/weather"

        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.weather_api_key,
            'units': 'metric'
        }

        response = requests.get(base_url, params=params)
        response.raise_for_status()

        data = response.json()
        return {
            'temperature': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'wind_speed': data['wind']['speed'],
            'pressure': data['main']['pressure']
        }

    def _collect_elevation(self, lat, lon):
        payload = {
            "locations": [{"latitude": lat, "longitude": lon}]
        }
        response = requests.post(self.elevation_api, json=payload)
        response.raise_for_status()

        data = response.json()
        return {
            'elevation': data['results'][0]['elevation']
        }

    def _collect_firms_data(self, lat, lon):
        return {
            'firms_confidence': 0,
            'firms_brightness': 0
        }

    def _collect_vegetation(self, lat, lon):
        """
        Dummy vegetation type assignment (placeholder for now)
        """
        import random
        vegetation_types = ['Forest', 'Shrubland', 'Grassland', 'Urban', 'Water', 'Barren']
        veg = random.choice(vegetation_types)
        return {'vegetation_type': veg}


def main():
    collector = AdvancedDataCollector()

    locations = [
        (40.7128, -74.0060),  # New York
        (34.0522, -118.2437),  # Los Angeles
        (51.5074, -0.1278),  # London
    ]

    data = collector.collect_comprehensive_data(locations)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data.to_csv(f'fire_training_data_real_{timestamp}.csv', index=False)

    print("✅ Data collection complete!")


if __name__ == "__main__":
    main()
