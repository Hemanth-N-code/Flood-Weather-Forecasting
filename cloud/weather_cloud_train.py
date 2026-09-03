import xarray as xr
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

print("Loading ERA5 weather datasets...")

instant_ds = xr.open_dataset("data/data_stream-oper_stepType-instant.nc")
accum_ds = xr.open_dataset("data/data_stream-oper_stepType-accum.nc")

instant_df = instant_ds.to_dataframe().reset_index()
accum_df = accum_ds.to_dataframe().reset_index()

print("Merging datasets...")

df = pd.merge(
    instant_df,
    accum_df,
    on=["valid_time","latitude","longitude"]
)

print("Creating wind speed...")

df["wind_speed"] = np.sqrt(
    df["u10"]**2 + df["v10"]**2
)

print("Renaming ERA5 variables...")

df = df.rename(columns={
    "t2m":"temperature_c",
    "d2m":"dewpoint_c",
    "msl":"mean_sea_level_pressure_hpa",
    "sp":"surface_pressure_hpa",
    "tp":"rainfall_mm"
})
# ===============================
# 🔥 UNIT CONVERSION (CRITICAL FIX)
# ===============================
print("Converting units...")

# Kelvin → Celsius
df["temperature_c"] = df["temperature_c"] - 273.15
df["dewpoint_c"] = df["dewpoint_c"] - 273.15

# meters → millimeters
df["rainfall_mm"] = df["rainfall_mm"] * 1000
print("Creating time features...")

df["datetime"] = pd.to_datetime(df["valid_time"])
df["hour"] = df["datetime"].dt.hour

df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)

features = [
    "dewpoint_c",
    "temperature_c",
    "mean_sea_level_pressure_hpa",
    "surface_pressure_hpa",
    "rainfall_mm",
    "wind_speed",
    "hour_sin",
    "hour_cos"
]

print("Preparing dataset...")

data = df[features].dropna()

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# -----------------------------
# Create sequences for LSTM
# -----------------------------

def create_sequences(data, seq_length=24):

    X = []
    y = []

    for i in range(len(data)-seq_length):

        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])

    return np.array(X), np.array(y)


SEQ_LENGTH = 24

X, y = create_sequences(scaled_data, SEQ_LENGTH)

print("Sequence shape:", X.shape)

# Train-test split

split = int(0.8 * len(X))

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

print("Building LSTM model...")

model = Sequential()

model.add(LSTM(64, return_sequences=True, input_shape=(SEQ_LENGTH, len(features))))
model.add(Dropout(0.2))

model.add(LSTM(64))
model.add(Dropout(0.2))

model.add(Dense(len(features)))

model.compile(
    optimizer="adam",
    loss="mse"
)

print(model.summary())

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

print("Training LSTM model...")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test,y_test),
    epochs=30,
    batch_size=64,
    callbacks=[early_stop]
)

print("Evaluating model...")

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)

os.makedirs("../models",exist_ok=True)

print("Saving model...")

model.save("../models/weather_lstm_model.h5")
joblib.dump(scaler,"../models/weather_scaler.pkl")

with open("../models/weather_metrics.txt","w") as f:
    f.write(f"MAE: {mae}\n")
    f.write(f"RMSE: {rmse}\n")
    f.write(f"R2: {r2}\n")

print("Generating plots...")

plt.plot(history.history["loss"],label="train")
plt.plot(history.history["val_loss"],label="validation")
plt.legend()
plt.title("Training Loss")
plt.savefig("../models/training_loss.png")
plt.close()

print("Weather LSTM training complete.")