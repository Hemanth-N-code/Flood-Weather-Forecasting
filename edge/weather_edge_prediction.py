# ===============================
# 1. INSTALL & IMPORTS
# ===============================
import firebase_admin
from firebase_admin import credentials, storage, firestore
import requests
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import time
import os
from datetime import datetime, timezone, timedelta
IST = timezone(timedelta(hours=5, minutes=30))

# ===============================
# 2. LOCATIONS (All Verified Assam Districts)
# ===============================
ASSAM_LOCATIONS = {
    "Dibrugarh": {"lat": 27.4728, "lon": 94.9120},
    "Silchar": {"lat": 24.8333, "lon": 92.7789},
    "Tezpur": {"lat": 26.6528, "lon": 92.7926},
    "Jorhat": {"lat": 26.7509, "lon": 94.2037},
    "Nagaon": {"lat": 26.3480, "lon": 92.6830},
    "Bongaigaon": {"lat": 26.4760, "lon": 90.5582},
    "Tinsukia": {"lat": 27.4886, "lon": 95.3558},
    "Dhubri": {"lat": 26.0207, "lon": 89.9743},
    "Diphu": {"lat": 25.8450, "lon": 93.4294},
    "Goalpara": {"lat": 26.1770, "lon": 90.6260},
    "Barpeta": {"lat": 26.3212, "lon": 91.0066},
    "Morigaon": {"lat": 26.2500, "lon": 92.3400},
    "Guwahati": {"lat": 26.1445, "lon": 91.7362}
}

FEATURES = [
    "dewpoint_c",
    "temperature_c",
    "mean_sea_level_pressure_hpa",
    "surface_pressure_hpa",
    "rainfall_mm",
    "wind_speed",
    "hour_sin",
    "hour_cos"
]

def init_firebase():
    """Safely initialize Firebase using local service account."""
    if not firebase_admin._apps:
        cred_path = "your_service_account.json"
        if not os.path.exists(cred_path):
            # Check edge folder relative to root
            cred_path = os.path.join(os.path.dirname(__file__), "your_service_account.json")
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred, {
            'storageBucket': 'flood-weather-app.firebasestorage.app'
        })
    return firestore.client(), storage.bucket()

def load_weather_assets(bucket=None):
    """Load model and scaler from local cache or download from Firebase Storage."""
    base_dir = os.path.dirname(__file__)
    local_model = os.path.join(base_dir, "weather_lstm_model.h5")
    local_scaler = os.path.join(base_dir, "weather_scaler.pkl")
    root_model = os.path.join(base_dir, "..", "models", "weather_lstm_model.h5")
    root_scaler = os.path.join(base_dir, "..", "models", "weather_scaler.pkl")

    if not os.path.exists(local_model) and os.path.exists(root_model):
        import shutil
        shutil.copy(root_model, local_model)
    if not os.path.exists(local_scaler) and os.path.exists(root_scaler):
        import shutil
        shutil.copy(root_scaler, local_scaler)

    # Fallback to downloading if still not present
    if (not os.path.exists(local_model) or not os.path.exists(local_scaler)) and bucket:
        print("📥 Downloading weather assets from Cloud Storage...")
        bucket.blob("models/weather_lstm_model.h5").download_to_filename(local_model)
        bucket.blob("models/weather_scaler.pkl").download_to_filename(local_scaler)

    model = tf.keras.models.load_model(local_model, compile=False)
    scaler = joblib.load(local_scaler)
    return model, scaler

def run_weather_prediction():
    """Main execution function for real-time weather predictions across Assam."""
    db, bucket = init_firebase()
    model, scaler = load_weather_assets(bucket)
    weather_ref = db.collection("weather_forecasts")

    print("\n🚀 Live Assam Weather Prediction Started\n")
    current_time_sync = datetime.now(IST).strftime("%Y-%m-%d %I:%M:%S %p IST")
    results = {}

    for place, coords in ASSAM_LOCATIONS.items():
        lat, lon = coords["lat"], coords["lon"]
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&past_days=1&forecast_days=1&"
            f"hourly=temperature_2m,dew_point_2m,surface_pressure,pressure_msl,rain,wind_speed_10m"
        )

        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"⚠️ Failed to fetch live data for {place}: {e}. Retrying once...")
            time.sleep(1)
            resp = requests.get(url, timeout=20)
            data = resp.json()

        df = pd.DataFrame(data['hourly'])

        # Select last 24 hours
        df_last_24 = df.tail(24).copy()

        # Rename columns to match model features
        df_last_24 = df_last_24.rename(columns={
            "temperature_2m": "temperature_c",
            "dew_point_2m": "dewpoint_c",
            "pressure_msl": "mean_sea_level_pressure_hpa",
            "surface_pressure": "surface_pressure_hpa",
            "rain": "rainfall_mm",
            "wind_speed_10m": "wind_speed"
        })

        # Time cyclical features
        df_last_24["datetime"] = pd.to_datetime(df_last_24["time"])
        df_last_24["hour"] = df_last_24["datetime"].dt.hour
        df_last_24["hour_sin"] = np.sin(2 * np.pi * df_last_24["hour"] / 24)
        df_last_24["hour_cos"] = np.cos(2 * np.pi * df_last_24["hour"] / 24)

        # Handle missing values cleanly
        df_last_24 = df_last_24.ffill().bfill()

        model_input_df = df_last_24[FEATURES]

        # Scale features
        scaled_input = scaler.transform(model_input_df)
        current_sequence = scaled_input.reshape((1, 24, len(FEATURES)))

        # Predict next 1 hour
        prediction_scaled = model.predict(current_sequence, verbose=0)
        forecast = scaler.inverse_transform(prediction_scaled)[0]

        temp = float(round(forecast[1], 1))
        rain = float(round(max(0, forecast[4]), 2))
        wind = float(round(forecast[5], 1))

        doc_data = {
            "place_name": place,
            "latitude": float(lat),
            "longitude": float(lon),
            "updated_at": current_time_sync,
            "next_hour_forecast": {
                "temp_c": temp,
                "rain_mm": rain,
                "wind_ms": wind
            }
        }

        weather_ref.document(place).set(doc_data)
        results[place] = doc_data["next_hour_forecast"]
        print(f"🌍 {place:<12} → Temp: {temp:>4.1f}°C | Rain: {rain:>4.2f}mm | Wind: {wind:>3.1f}m/s")
        time.sleep(0.3)

    print(f"\n✅ Weather Sync Complete: {len(results)} stations updated at {current_time_sync}\n")
    return results

if __name__ == "__main__":
    run_weather_prediction()