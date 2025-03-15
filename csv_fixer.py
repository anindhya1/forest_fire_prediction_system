# csv_fixer.py
import os
import pandas as pd

# Path to the CSV file
csv_file = 'fire_data.csv'

# Check if file exists but is empty
if os.path.exists(csv_file) and os.path.getsize(csv_file) == 0:
    print(f"CSV file exists but is empty. Adding headers...")

    # Create DataFrame with headers
    headers = ['timestamp', 'temperature', 'humidity', 'wind_speed',
               'rainfall', 'days_without_rain', 'risk_level']
    df = pd.DataFrame(columns=headers)

    # Write headers to CSV
    df.to_csv(csv_file, index=False)
    print(f"Headers added to {csv_file} successfully.")
else:
    print(f"CSV file checking...")
    try:
        # Try reading to see if it has proper headers
        df = pd.read_csv(csv_file)
        print(f"CSV file exists and has columns: {list(df.columns)}")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        # Create a new file with headers
        headers = ['timestamp', 'temperature', 'humidity', 'wind_speed',
                   'rainfall', 'days_without_rain', 'risk_level']
        df = pd.DataFrame(columns=headers)
        df.to_csv(csv_file, index=False)
        print(f"Created new CSV file with headers.")

print("Done!")