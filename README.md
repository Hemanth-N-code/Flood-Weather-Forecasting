# 🛰️ EdgeAI Sentinel: Spatio-Temporal Flood Inundation & Meteorological Forecasting System

[![Live Demo](https://img.shields.io/badge/Live_Dashboard-flood--weather--app.web.app-0ea5e9?style=for-the-badge&logo=google-chrome&logoColor=white)](https://flood-weather-app.web.app)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Firebase](https://img.shields.io/badge/Firebase-Firestore_%26_Hosting-FFCA28?style=for-the-badge&logo=firebase&logoColor=black)](https://firebase.google.com)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

An end-to-end, edge-deployable deep learning framework for **real-time flood risk mapping and short-term weather forecasting** across the floodplains of the **Brahmaputra and Barak basins in Assam, India**.

---

## 📌 Key Highlights

* 🌐 **Live Public Web Dashboard**: Hosted at **[https://flood-weather-app.web.app](https://flood-weather-app.web.app)** with interactive GIS Leaflet maps and 2-to-3 decimal precision telemetry.
* 🌊 **ConvLSTM2D-UNet Inundation Head**: Ingests multi-modal 6-channel spatio-temporal tensors $(N, 5, 32, 32, 6)$ to predict local flood probability with **97.0% overall accuracy** and **99.0% flood recall**.
* ⛅ **Sequence-to-Sequence Weather LSTM**: Models 24-hour non-linear atmospheric dynamics across 8 thermodynamic features to forecast next-hour conditions ($R^2 = 0.9352$, $\text{MAE} = 0.0975$).
* ⚡ **Sub-Minute Edge Latency**: Replaces computationally heavy 2D numerical hydrodynamic solvers (which take hours) with neural inference that executes in **$\approx 61\text{ seconds}$**.
* 🔬 **Hydrologically Validated**: Statistically proven physical consistency with $+0.8189$ rainfall correlation and $-0.5144$ elevation correlation.

---

## 📊 Model Performance & Validation Benchmarks

### 1. Flood Inundation Model (ConvLSTM2D + UNet Hybrid)
Evaluated across 929 independent spatio-temporal validation rasters:

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **0: Non-Flooded Land** | `0.99` | `0.95` | `0.97` | 475 |
| **1: Inundated Flood Water** | `0.95` | **`0.99`** | `0.97` | 454 |
| **Overall Accuracy** | — | — | **`97.0%`** | 929 |
| **Macro / Weighted Avg** | `0.97` | `0.97` | `0.97` | 929 |

> **Hazard Mitigation Impact**: Achieving a **99.0% recall on the flood class** ensures near-zero false negatives, making the framework dependable for emergency early warnings.

### 2. Meteorological Forecasting Model (Sequence-to-Sequence LSTM)
Evaluated on continuous multi-month hourly atmospheric test sequences:

| Metric | Empirical Score | Benchmark Target | Description |
| :--- | :---: | :---: | :--- |
| **Coefficient of Determination ($R^2$)** | **`0.9352`** | $> 0.85$ | Explains **93.52% of total temporal variance** |
| **Mean Absolute Error (MAE)** | **`0.0975`** | $< 0.15$ | Low normalized error across all 8 features |
| **Root Mean Square Error (RMSE)** | **`0.1955`** | $< 0.25$ | Robust against transient convective shocks |

---

## 🗺️ System Architecture & Data Flow

```
[ Tier 1: Real-Time Data Ingestion ]
   ├── Open-Meteo REST API: Live 24-hour meteorological observations
   ├── Hydrological API: Live 5-day cumulative precipitation history
   └── Geospatial Vectors: SRTM 30m DEM Elevation, Terrain Slope, River Proximity
                  │
                  ▼
[ Tier 2: Autonomous EdgeAI Sentinel (run_live_sentinel.py) ]
   ├── weather_edge_prediction.py  ──> LSTM (1, 24, 8)       ──> Next-Hour Weather
   └── flood_edge_prediction.py    ──> ConvLSTM2D (N, 5, 32, 32, 6) ──> Spatial Flood Risk
                  │
                  ▼
[ Tier 3: Cloud Database Synchronization (Firebase) ]
   ├── Cloud Firestore: 'weather_forecasts' (13 Assam stations)
   ├── Cloud Firestore: 'predictions'       (16 Assam hydrological zones)
   └── Cloud Storage:   Model checkpoints (.keras, .h5, .pkl) & npz tensors
                  │
                  ▼
[ Tier 4: Client Presentation Layer (Firebase Hosting) ]
   └── app/index.html (Leaflet GIS, CORS-Free Compat Architecture, Auto-polling)
```

---

## 📁 Repository Structure

```
Flood-Weather-Forecasting/
├── app/
│   └── index.html                 # Production GIS dashboard (Leaflet + Firebase Compat)
├── cloud/
│   ├── flood_cloud_train.py       # Cloud training pipeline for ConvLSTM2D-UNet
│   ├── flood_data_extract.py      # Earth Engine multi-spectral raster extractor
│   ├── flood_preprocess_upload.py # Patch slicer & tensor generation
│   ├── flood_upload.py            # Model checkpoint uploader
│   └── weather_cloud_train.py     # Training pipeline for Sequence-to-Sequence LSTM
├── data/
│   ├── data_stream-oper_stepType-accum.nc   # ECMWF ERA5 precipitation reanalysis
│   └── data_stream-oper_stepType-instant.nc # ECMWF ERA5 thermodynamic reanalysis
├── edge/
│   ├── downloaded_data.npz        # Preprocessed genuine live tensor
│   ├── downloaded_model.keras     # Local checkpoint of ConvLSTM2D-UNet model
│   ├── flood_edge_prediction.py   # Live flood prediction engine
│   ├── service_account.sample.json# Template for Firebase credentials
│   ├── weather_edge_prediction.py # Live weather prediction engine
│   ├── weather_lstm_model.h5      # Local checkpoint of weather LSTM model
│   └── weather_scaler.pkl         # Fitted StandardScaler
├── models/
│   ├── accuracy.png               # Training accuracy curve
│   ├── classification_report.txt  # Quantitative validation metrics
│   ├── confusion_matrix.png       # Model confusion matrix
│   ├── flood_convlstm_model.keras # Master checkpoint
│   ├── loss.png                   # Training loss curve
│   ├── prediction_distribution.png# Probability density
│   ├── weather_lstm_model.h5      # Master weather model
│   └── weather_metrics.txt        # MAE, RMSE, and R2 scores
├── firebase.json                  # Firebase Hosting configuration
├── .firebaserc                    # Firebase project configuration
├── JOURNAL_PAPER_TECHNICAL_REPORT.md # Academic manuscript for journal submission
├── requirements.txt               # Python package dependencies
├── run_live_sentinel.py           # Master automated real-time prediction runner
└── start_sentinel_service.bat     # Windows 1-click 24/7 background daemon launcher
```

---

## 🚀 Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Hemanth-N-code/Flood-Weather-Forecasting.git
cd Flood-Weather-Forecasting
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv venv

# On Windows:
.\venv\Scripts\activate

# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure Credentials
Place your Firebase Admin SDK service account key JSON inside the `edge/` directory as `your_service_account.json` (see `edge/service_account.sample.json` for structure).

### 4. Execute Real-Time Predictions
```bash
# Run both models once:
python run_live_sentinel.py

# Run as an autonomous continuous background daemon (every 60 minutes):
python run_live_sentinel.py --daemon --interval 60
```

### 5. Launch the Client Dashboard
* **Option A**: Visit the live website at **[https://flood-weather-app.web.app](https://flood-weather-app.web.app)**
* **Option B**: Open `app/index.html` directly in any web browser.

---

## 📄 Academic Publication & Journal Manuscript
A full technical manuscript formatted for peer-reviewed journal submission (IEEE TGRS / Elsevier / Springer) is provided in:
👉 **[`JOURNAL_PAPER_TECHNICAL_REPORT.md`](JOURNAL_PAPER_TECHNICAL_REPORT.md)**

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
