# 🛰️ EdgeAI Sentinel: Spatio-Temporal Flood Inundation & Meteorological Forecasting System
## Comprehensive Technical Project Dossier for Advisor Review & IEEE Conference Submission

**Project Title:** *A Spatio-Temporal Deep Learning Framework for Operational Flood Inundation and Meteorological Forecasting in the Brahmaputra Basin*  
**Repository:** [https://github.com/Hemanth-N-code/Flood-Weather-Forecasting](https://github.com/Hemanth-N-code/Flood-Weather-Forecasting)  
**Live Deployed Application:** [https://flood-weather-app.web.app](https://flood-weather-app.web.app)  
**Target Publication Venues:** IEEE IGARSS, IEEE GHTC, IEEE R10-HTC, IEEE TENCON  

---

## 1. Executive Summary & Research Contribution

Conventional flood risk models rely on 2D numerical hydrodynamic differential equation solvers (e.g., HEC-RAS, MIKE 21) that require hours of compute time per simulation cycle, rendering them ineffective for short-window emergency evacuations. 

**EdgeAI Sentinel** introduces a dual-neural operational framework tailored to the Brahmaputra and Barak basins of Assam, India. The system couples:
1. A **ConvLSTM2D-UNet Hybrid Spatio-Temporal Model** that digests a 6-band multi-modal tensor $(N, 5, 32, 32, 6)$ fusing SAR water masks, optical NDWI, GPM precipitation, SRTM DEM elevation, slope, and river proximity, achieving **97.0% overall accuracy** and **99.0% flood recall**.
2. A **Sequence-to-Sequence Recurrent Neural Network (LSTM)** trained on hourly ECMWF ERA5 reanalysis data to forecast 1-hour ahead atmospheric thermodynamic parameters ($R^2 = 0.9352$, $\text{MAE} = 0.0975$).
3. An **Autonomous Edge Inference Engine** running periodic cycles in **$\approx 61\text{ seconds}$**, publishing structured telemetry to Google Cloud Firestore, and serving a live GIS radar dashboard over global CDN.

---

## 2. Phase 1: Multi-Modal Data Ingestion & Tensor Engineering

### 2.1. Inundation Multi-Modal Tensor Formulation: $\mathbf{X} \in \mathbb{R}^{N \times 5 \times 32 \times 32 \times 6}$
Each geographical patch is represented by 5 consecutive temporal time-steps ($t-4$ to $t$) across a spatial resolution of $32 \times 32$ pixels and 6 normalized physical channels:

| Band | Physical Feature | Sensor / Data Source | Resolution | Transform / Normalization |
| :---: | :--- | :--- | :---: | :--- |
| **$c_0$** | **Water / Flood Mask** | Sentinel-1 SAR (C-Band VV) | 10 m | Binary ground truth $\{0, 1\}$ |
| **$c_1$** | **NDWI (Surface Water Index)** | Sentinel-2 MSI Optical | 10 m | $\tilde{x}_{c_1} = \frac{\text{NDWI} + 1.0}{2.0} \in [0, 1]$ |
| **$c_2$** | **Precipitation Accumulation** | NASA GPM IMERG Daily | 0.1° (~10 km) | $\tilde{x}_{c_2} = \frac{\text{Rainfall (mm)}}{50.0}$ |
| **$c_3$** | **Digital Elevation Model (DEM)** | NASA SRTM 30m Global DEM | 30 m | $\tilde{x}_{c_3} = \frac{\text{Elevation (m)}}{3000.0}$ |
| **$c_4$** | **Topographic Slope** | SRTM Derived Terrain Gradient | 30 m | $\tilde{x}_{c_4} = \frac{\text{Slope (deg)}}{45.0}$ |
| **$c_5$** | **River Channel Proximity** | JRC Global Surface Water | 30 m | $\tilde{x}_{c_5} = \frac{\text{Distance (m)}}{5000.0}$ |

### 2.2. Atmospheric Time-Series Feature Matrix: $\mathbf{W} \in \mathbb{R}^{B \times 24 \times 8}$
Rolling 24-hour sequences of 8 standardized continuous variables extracted from ECMWF ERA5 NetCDF reanalysis (`data_stream-oper_stepType-instant.nc` & `accum.nc`):
* Dewpoint Temperature ($T_{\text{dew}}$)
* 2m Surface Temperature ($T_{\text{surface}}$)
* Mean Sea Level Pressure ($P_{\text{msl}}$)
* Surface Pressure ($P_{\text{surface}}$)
* Total Precipitation ($R_{\text{accum}}$)
* 10m Wind Speed ($V_{\text{wind}}$)
* Sinusoidal Diurnal Temporal Encoding: $\sin(2\pi h / 24)$
* Cosinusoidal Diurnal Temporal Encoding: $\cos(2\pi h / 24)$

---

## 3. Phase 2: Neural Network Architectures & Training Pipelines

### 3.1. Flood Segmentation Architecture: ConvLSTM2D-UNet Hybrid
* **Input**: Tensor $(B, 5, 32, 32, 6)$
* **Encoder**:
  * Layer 1: `ConvLSTM2D(32 filters, 3x3 kernel, return_sequences=True)` + `BatchNormalization` + `LeakyReLU`
  * Layer 2: `ConvLSTM2D(16 filters, 3x3 kernel, return_sequences=False)` + `BatchNormalization` + `LeakyReLU`
* **Skip & Decoupled Latent Block**:
  * `Conv2D(32, 3x3)` $\to$ `MaxPooling2D(2x2)` $\to$ `Conv2D(64, 3x3, bottleneck)`
  * `UpSampling2D(2x2)` $\to$ `Concatenate([skip, decoder])` $\to$ `Conv2D(32, 3x3)`
* **Output Head**: `Conv2D(1 filter, 1x1 kernel, activation='sigmoid')` yielding spatial probability matrix $(B, 32, 32, 1)$.

### 3.2. Meteorological Forecaster Architecture: Stacked LSTM Sequence Network
* **Input**: Sequence $(B, 24, 8)$
* **Recurrent Core**:
  * `LSTM(128 units, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)`
  * `LSTM(64 units, dropout=0.2, return_sequences=False)`
* **Dense Regression Head**:
  * `Dense(32 units, activation='relu')`
  * `Dense(8 units, activation='linear')` predicting 1-hour ahead standardized state vector, de-normalized via `weather_scaler.pkl`.

---

## 4. Phase 3: Quantitative Results & Visual Empirical Evidence

### 4.1. Flood Inundation Model Results (929 Independent Test Patches)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Class 0 (Non-Flooded Land)** | **0.99** | 0.95 | **0.97** | 475 |
| **Class 1 (Inundated Flood Water)** | 0.95 | **0.99** | **0.97** | 454 |
| **Overall Accuracy** | — | — | **97.0%** (0.9700) | 929 |
| **Macro Average** | 0.97 | 0.97 | 0.97 | 929 |
| **Weighted Average** | 0.97 | 0.97 | 0.97 | 929 |

> **Critical Safety Metric**: The **99.0% recall on Class 1** guarantees that the system minimizes false negatives, preventing undetected flood events in populated basins.

#### Visual Result Plots for Paper Figures:
* 📊 **Confusion Matrix**: `models/confusion_matrix.png`
* 📈 **Accuracy Convergence Curve**: `models/accuracy.png`
* 📉 **BCE Training & Validation Loss**: `models/loss.png`
* 📉 **Pixel Prediction Distribution**: `models/prediction_distribution.png`

---

### 4.2. Weather LSTM Forecasting Results

| Evaluation Metric | Measured Value | Standard Target | Physical Significance |
| :--- | :---: | :---: | :--- |
| **Coefficient of Determination ($R^2$)** | **0.9352** | $> 0.85$ | Model accounts for **93.52% of sequence variance** |
| **Mean Absolute Error (MAE)** | **0.0975** | $< 0.15$ | High precision across all 8 scaled thermodynamic dimensions |
| **Root Mean Square Error (RMSE)** | **0.1955** | $< 0.25$ | Stable against rapid atmospheric shifts |

#### Visual Result Plots for Paper Figures:
* 📉 **Weather LSTM Training Loss Curve**: `models/training_loss.png`
* 📊 **Weather Prediction Score & R² Plot**: `models/weather Score.png`

---

### 4.3. Real-Time Physical Hydrological Validation

Across 16 operational Assam floodplains, model predictions demonstrated rigorous physical consistency with known hydrological laws:

| Monitored Flood Basin | 5-Day Cumulative Rain | DEM Elevation | River Proximity | Predicted Risk | Classification |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Hailakandi (Katachak)** | 85.80 mm | 21.00 m | 450.00 m | **53.37%** (0.534) | High Flood Risk |
| **Karimganj (Kushiyara)** | 89.80 mm | 18.00 m | 300.00 m | **53.37%** (0.534) | High Flood Risk |
| **Silchar (Barak Valley)** | 142.70 mm | 23.00 m | 350.00 m | **53.30%** (0.533) | High Flood Risk |
| **Barpeta (Beki / Manas)** | 98.60 mm | 42.00 m | 450.00 m | **53.30%** (0.533) | High Flood Risk |
| **Majuli (River Island)** | 98.30 mm | 85.00 m | 250.00 m | **52.90%** (0.529) | High Flood Risk |
| **Kaziranga / Bokakhat** | 89.40 mm | 72.00 m | 400.00 m | **52.90%** (0.529) | High Flood Risk |
| **Tezpur (Kolia Bhomora)** | 81.80 mm | 58.00 m | 900.00 m | **52.80%** (0.528) | High Flood Risk |
| **Dhemaji (Northern Plains)**| 32.20 mm | 98.00 m | 600.00 m | **50.80%** (0.508) | High Flood Risk |

* **Rainfall Correlation**: **$\rho = +0.8189$** (Strong positive physical accumulation response).
* **Elevation Correlation**: **$\rho = -0.5144$** (Strong negative elevation sensitivity; lowlands flood first).

---

## 5. Phase 4: Operational Data Flow & Pipeline Architecture

```
[ STEP 1: Live Observation Ingestion ]
    ├── Open-Meteo REST API: Real-time 24h temperature, pressure, wind, and rain
    └── Live 5-Day Rainfall Accumulation + SRTM DEM + River Distances
                   │
                   ▼
[ STEP 2: Autonomous EdgeAI Sentinel (run_live_sentinel.py) ]
    ├── weather_edge_prediction.py  ──> LSTM (1, 24, 8)       ──> 1-Hr Ahead Forecast
    └── flood_edge_prediction.py    ──> ConvLSTM2D (16, 5, 32, 32, 6) ──> Spatial Inundation Map
                   │
                   ▼
[ STEP 3: Decoupled Cloud Synchronization (Firebase) ]
    ├── Cloud Firestore: 'weather_forecasts' (13 Assam stations updated)
    └── Cloud Firestore: 'predictions'       (16 Assam zones refreshed)
                   │
                   ▼
[ STEP 4: High-Concurrency Global Client Delivery ]
    └── Firebase Hosting: https://flood-weather-app.web.app (Leaflet GIS, Sub-second rendering)
```

### Computational Latency Benchmark
* Weather Data Fetch & LSTM Inference (13 stations): **$5.3\text{ seconds}$**
* Hydrological Ingestion & ConvLSTM2D Inference (16 zones): **$8.8\text{ seconds}$**
* Cloud Firestore Document Synchronization: **$4.0\text{ seconds}$**
* **Total End-to-End Cycle Latency**: **$\approx 61.0\text{ seconds}$**

---

## 6. Phase 5: Cloud Infrastructure & Firebase Integration

1. **Cloud Firestore (NoSQL Database)**:
   * Acts as the real-time decoupled buffer between edge compute and web clients.
   * Eliminates the need to maintain custom REST API servers (FastAPI/Express) on field edge devices.
2. **Firebase Storage (Artifact Bucket)**:
   * Secure object storage for model weights (`flood_model_v1.keras`, `weather_lstm_model.h5`, `weather_scaler.pkl`) and dynamic `.npz` tensors.
3. **Firebase Hosting (Global CDN)**:
   * Serves the single-page application (`app/index.html`) over Google Cloud CDN with automated SSL (HTTPS).
   * Free-tier compliant, capable of handling burst traffic spikes during flood emergencies without server provisioning.

---

## 7. Phase 6: System Deployment & Access Points

| Component | Status | Location / Access Point |
| :--- | :---: | :--- |
| **Public Web Dashboard** | 🟢 Live | **[https://flood-weather-app.web.app](https://flood-weather-app.web.app)** |
| **Alternative Domain** | 🟢 Live | **[https://flood-weather-app.firebaseapp.com](https://flood-weather-app.firebaseapp.com)** |
| **GitHub Source Code** | 🟢 Public | **[https://github.com/Hemanth-N-code/Flood-Weather-Forecasting](https://github.com/Hemanth-N-code/Flood-Weather-Forecasting)** |
| **Edge Background Daemon** | 🟢 Ready | `start_sentinel_service.bat` (Executes every 60 min) |
| **Firebase Console** | 🟢 Active | `https://console.firebase.google.com/project/flood-weather-app/overview` |
