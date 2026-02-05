import pandas as pd
import joblib
from pymongo import MongoClient
from datetime import timedelta
import os

HOURS_AHEAD = 72

# -------------------------
# FETCH LATEST FEATURE FROM MONGO
# -------------------------
def load_latest_features():
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise ValueError("❌ MONGO_URI environment variable is not set")

    client = MongoClient(mongo_uri)
    col = client["Pearls_aqi_feature_store"]["karachi_air_quality_index"]
    df = pd.DataFrame(list(col.find({}, {"_id": 0})))

    if df.empty:
        raise ValueError("❌ No data found in MongoDB feature store")

    # Convert timestamp and sort
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    return df.iloc[-1:].copy()  # latest row only

# -------------------------
# LOAD LATEST MODEL FROM LOCAL
# -------------------------
def load_latest_model():
    model_folder = "models"
    if not os.path.exists(model_folder):
        raise FileNotFoundError("❌ Models folder not found")

    all_models = [f for f in os.listdir(model_folder) if f.endswith(".pkl")]
    if not all_models:
        raise FileNotFoundError("❌ No model (.pkl) found in models folder")

    # Sort by modified time so latest saved model is used
    latest_model = max(all_models, key=lambda x: os.path.getmtime(os.path.join(model_folder, x)))
    model_path = os.path.join(model_folder, latest_model)
    print(f"✅ Loading model: {model_path}")

    return joblib.load(model_path)
    

# -------------------------
# FORECAST 3 DAYS / 72 HOURS
# -------------------------
def forecast_3_days():
    model = load_latest_model()
    current = load_latest_features()

    predictions = []

    for step in range(HOURS_AHEAD):
        X = current.drop(columns=["timestamp", "location", "pm25_next_hour"], errors="ignore")
        pred_pm25 = model.predict(X)[0]
        predictions.append(pred_pm25)

        # Advance time
        next_time = current["timestamp"].iloc[0] + timedelta(hours=1)

        # Update lag features
        if "pm25_lag_1h" in current.columns:
            current["pm25_lag_6h"] = current.get("pm25_lag_3h", pred_pm25)
            current["pm25_lag_3h"] = current.get("pm25_lag_1h", pred_pm25)
        current["pm25_lag_1h"] = pred_pm25

        # Update rolling value
        current["pm25"] = pred_pm25

        # Update timestamp & time features
        current["timestamp"] = next_time
        current["hour"] = next_time.hour
        current["day"] = next_time.day
        current["month"] = next_time.month
        current["day_of_week"] = next_time.dayofweek
        current["is_weekend"] = int(next_time.dayofweek >= 5)
        current["is_rush_hour"] = int(next_time.hour in [7,8,9,17,18,19])

    return predictions

if __name__ == "__main__":
    preds = forecast_3_days()
    print("✅ 3-Day / 72h AQI Forecast Generated")
