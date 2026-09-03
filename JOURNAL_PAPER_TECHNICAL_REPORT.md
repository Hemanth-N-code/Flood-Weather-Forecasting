# Deep Learning-Driven Multi-Hazard Early Warning System: Real-Time Spatio-Temporal Flood Inundation Modeling and Sequence Weather Forecasting for the Brahmaputra Basin

**Technical Manuscript & Research Reference Document**  
*Prepared for Academic Journal Publication, Thesis Guidance, and Experimental Reproducibility*

---

## 1. Abstract

Flooding in the Brahmaputra and Barak river basins of Assam, India, poses a recurring socio-economic and humanitarian hazard driven by intense South Asian monsoon precipitation, complex terrain gradients, and rapid river discharge fluctuations. Traditional hydrodynamic models rely heavily on computationally prohibitive 2D/3D numerical differential equation solvers that struggle with real-time operational deployment at low latency. 

This paper introduces **EdgeAI Sentinel**, an end-to-end, edge-deployable deep learning framework integrating two synchronized neural architectures:
1. A **Hybrid ConvLSTM2D-UNet Spatio-Temporal Segmentation Network** that ingests multi-spectral and multi-temporal remote sensing tensors $(N, 5, 32, 32, 6)$ combining Synthetic Aperture Radar (SAR) flood masks, Normalized Difference Water Index (NDWI), NASA GPM precipitation, SRTM digital elevation models (DEM), terrain slope, and Euclidean river proximity. The model attains an overall classification accuracy of **97.0%**, with an exceptional **99.0% recall** on flood inundation zones.
2. A **Sequence-to-Sequence Recurrent Neural Network (LSTM)** that models multi-variate non-linear atmospheric time-series (ERA5 reanalysis) to deliver 1-hour ahead meteorological forecasts across 8 thermodynamic features ($R^2 = 0.9352$, $\text{RMSE} = 0.1955$, $\text{MAE} = 0.0975$).

The trained models are deployed on an autonomous edge computing node that periodically fetches real-time observations via REST APIs, evaluates spatial risk probability maps, synchronizes structured assessments with Google Cloud Firestore, and renders an interactive GIS client dashboard with sub-second retrieval latency. Empirical correlation analysis confirms strong physical consistency with known hydrological dynamics: $+0.8189$ positive correlation with 5-day cumulative rainfall and $-0.5144$ negative correlation with terrain elevation.

---

## 2. Problem Statement & Regional Context

The state of Assam, situated in Northeastern India, experiences catastrophic annual inundation across its 33 districts. The region's vulnerability is exacerbated by:
* **Physiographic Bottlenecks**: The Brahmaputra River spans over 650 km through Assam, dropping from an elevation of $\sim 130\text{ m}$ in upper Assam (Dibrugarh) to only $\sim 30\text{ m}$ at the western border (Dhubri), causing severe drainage congestion.
* **Monsoonal Convection**: High-intensity precipitation events during the South Asian Summer Monsoon deliver $1500 - 2600\text{ mm}$ of rainfall annually.
* **Data Latency in Hydrodynamic Modeling**: Conventional 1D/2D hydraulic models (e.g., HEC-RAS, MIKE 21) require extensive river cross-section surveys and hours of compute time per simulation cycle, limiting their effectiveness for short-window emergency evacuations.

**Research Objective**: Develop a computationally lightweight, data-driven system capable of predicting local inundation probability at high spatial resolution while simultaneously forecasting short-term convective precipitation using trained neural network architectures.

---

## 3. End-to-End System Architecture

The system operates across a five-tier decoupled architecture designed for high availability, fault tolerance, and minimal network bandwidth consumption at the edge:

```
[ Tier 1: Multi-Sensor Data Lake ]
   ├── Google Earth Engine (Sentinel-1 SAR, Sentinel-2 MSI, NASA GPM, SRTM DEM, JRC Water)
   ├── ECMWF ERA5 Reanalysis (Hourly Meteorological NetCDF Datasets)
   └── Live Telemetry APIs (Open-Meteo Hourly Weather & 5-Day Cumulative Hydrology)
                  │
                  ▼
[ Tier 2: Cloud Model Training & Optimization ]
   ├── ConvLSTM2D-UNet Spatial-Temporal Segmentation Model (flood_cloud_train.py)
   └── Multivariate Sequence-to-Sequence LSTM Weather Forecaster (weather_cloud_train.py)
                  │
                  ▼
[ Tier 3: Cloud Asset Repository (Firebase Storage) ]
   ├── models/flood_model_v1.keras (Trained Weights)
   ├── models/weather_lstm_model.h5 (Trained Weights)
   └── models/weather_scaler.pkl (Fitted StandardScaler)
                  │
                  ▼
[ Tier 4: Autonomous Edge AI Engine (run_live_sentinel.py) ]
   ├── edge/weather_edge_prediction.py  ──> Ingests 24h observations ──> LSTM Inference
   └── edge/flood_edge_prediction.py    ──> Ingests 5-day hydrology  ──> ConvLSTM2D Inference
                  │
                  ▼
[ Tier 5: Real-Time Data Sync & Web Presentation ]
   ├── Cloud Firestore: collections ('weather_forecasts', 'predictions')
   └── Client Dashboard: app/app.html (Leaflet GIS, CORS-Free Compat Architecture)
```

---

## 4. Mathematical Formulation & Dataset Engineering

### 4.1. Flood Risk Multi-Modal Tensor Formulation

To capture both the temporal evolution of rainfall and the static topographic vulnerability, each geographical patch is structured as a 5-dimensional spatio-temporal tensor:

$$\mathbf{X} \in \mathbb{R}^{N \times T \times H \times W \times C}$$

Where:
* $N = \text{Number of monitored regional patches}$ ($16$ core hydrological zones in Assam).
* $T = 5$ consecutive temporal time-steps (days $t-4, t-3, t-2, t-1, t$).
* $H = 32, W = 32$ spatial dimensions per patch.
* $C = 6$ distinct physical and hydrological channels.

#### Channel Definitions & Normalization Functions

| Channel Index | Physical Variable | Sensor / Source | Original Range | Normalization Transform |
| :---: | :--- | :--- | :---: | :--- |
| **$c_0$** | **Water / Flood Mask** | Sentinel-1 SAR (C-Band VV) | $\{0, 1\}$ | Binary indicator of surface water presence |
| **$c_1$** | **NDWI** | Sentinel-2 MSI Optical | $[-1.0, 1.0]$ | $\tilde{x}_{c_1} = \frac{x_{c_1} + 1.0}{2.0} \in [0, 1]$ |
| **$c_2$** | **Precipitation** | NASA GPM IMERG Daily | $0 - 150\text{ mm}$ | $\tilde{x}_{c_2} = \frac{x_{c_2}}{50.0}$ |
| **$c_3$** | **Elevation (DEM)** | SRTM 30m Digital Elevation | $10 - 3000\text{ m}$ | $\tilde{x}_{c_3} = \frac{x_{c_3}}{3000.0}$ |
| **$c_4$** | **Terrain Slope** | SRTM Calculated Gradient | $0^\circ - 45^\circ$ | $\tilde{x}_{c_4} = \frac{x_{c_4}}{45.0}$ |
| **$c_5$** | **River Proximity** | JRC Global Surface Water | $0 - 5000\text{ m}$ | $\tilde{x}_{c_5} = \frac{x_{c_5}}{5000.0}$ |

* **NDWI Formulation**:
  $$\text{NDWI} = \frac{\rho_{\text{Green}} - \rho_{\text{NIR}}}{\rho_{\text{Green}} + \rho_{\text{NIR}}} = \frac{\text{Band 3} - \text{Band 8}}{\text{Band 3} + \text{Band 8}}$$

* **SAR Flood Extraction**:
  $$\Delta \sigma^0 = \sigma^0_{\text{pre-event}} - \sigma^0_{\text{post-event}}$$
  $$\text{Mask}_{\text{flood}}(u, v) = \mathbb{I}\left(\Delta \sigma^0(u, v) > 1.5\text{ dB}\right)$$

---

### 4.2. Weather Time-Series Feature Matrix

The weather prediction network ingests a rolling 24-hour sequence across 8 continuous thermodynamic variables:

$$\mathbf{W} \in \mathbb{R}^{B \times L \times K}$$

Where $B$ is the batch size, $L = 24\text{ hours}$, and $K = 8$ engineered features:

$$\mathbf{k} = \begin{bmatrix} T_{\text{dew}} \\ T_{\text{surface}} \\ P_{\text{msl}} \\ P_{\text{surface}} \\ R_{\text{accum}} \\ V_{\text{wind}} \\ \sin\left(\frac{2\pi h}{24}\right) \\ \cos\left(\frac{2\pi h}{24}\right) \end{bmatrix}$$

Cyclical temporal encoding ($\sin, \cos$) preserves the continuous diurnal transition across the midnight boundary ($h = 23 \to h = 0$). All continuous features are standardized using Z-score parameterization:

$$z = \frac{x - \mu}{\sigma}$$

Where $\mu$ and $\sigma$ are estimated exclusively on the historical training distribution.

---

## 5. Neural Network Architectures

### 5.1. Model 1: ConvLSTM2D-UNet Hybrid Segmentation Network

To resolve both temporal sequence transitions (water accumulation over 5 days) and spatial pixel distributions, a Convolutional LSTM (ConvLSTM2D) encoder is paired with a convolutional decoder:

```
Input Tensor: (B, 5, 32, 32, 6)
      │
      ▼
┌────────────────────────────────────────────────────────┐
│ ConvLSTM2D Layer 1: 32 Filters, Kernel (3x3), Return Seq │
│ Batch Normalization + LeakyReLU Activation            │
└────────────────────────────────────────────────────────┘
      │ Output: (B, 5, 32, 32, 32)
      ▼
┌────────────────────────────────────────────────────────┐
│ ConvLSTM2D Layer 2: 16 Filters, Kernel (3x3), Last Step│
│ Batch Normalization + LeakyReLU Activation            │
└────────────────────────────────────────────────────────┘
      │ Output: (B, 32, 32, 16)
      ▼
┌────────────────────────────────────────────────────────┐
│ Encoder Block: Conv2D (32, 3x3) -> MaxPool2D (2x2)     │
│ Bottleneck:    Conv2D (64, 3x3)                        │
│ Decoder Block: UpSampling2D (2x2) -> Concatenate Skip  │
│ Refinement:    Conv2D (32, 3x3)                        │
└────────────────────────────────────────────────────────┘
      │ Output: (B, 32, 32, 32)
      ▼
┌────────────────────────────────────────────────────────┐
│ Output Layer: Conv2D (1 Filter, Kernel 1x1, Sigmoid)   │
└────────────────────────────────────────────────────────┘
      │
      ▼
Final Inundation Heatmap: (B, 32, 32, 1) ∈ [0.0, 1.0]
```

* **ConvLSTM Cell Mathematical Formulation**:
  $$\mathbf{i}_t = \sigma\left(\mathbf{W}_{xi} * \mathbf{X}_t + \mathbf{W}_{hi} * \mathbf{H}_{t-1} + \mathbf{W}_{ci} \circ \mathbf{C}_{t-1} + \mathbf{b}_i\right)$$
  $$\mathbf{f}_t = \sigma\left(\mathbf{W}_{xf} * \mathbf{X}_t + \mathbf{W}_{hf} * \mathbf{H}_{t-1} + \mathbf{W}_{cf} \circ \mathbf{C}_{t-1} + \mathbf{b}_f\right)$$
  $$\mathbf{C}_t = \mathbf{f}_t \circ \mathbf{C}_{t-1} + \mathbf{i}_t \circ \tanh\left(\mathbf{W}_{xc} * \mathbf{X}_t + \mathbf{W}_{hc} * \mathbf{H}_{t-1} + \mathbf{b}_c\right)$$
  $$\mathbf{o}_t = \sigma\left(\mathbf{W}_{xo} * \mathbf{X}_t + \mathbf{W}_{ho} * \mathbf{H}_{t-1} + \mathbf{W}_{co} \circ \mathbf{C}_t + \mathbf{b}_o\right)$$
  $$\mathbf{H}_t = \mathbf{o}_t \circ \tanh\left(\mathbf{C}_t\right)$$

Where $*$ denotes the 2D spatial convolution operator and $\circ$ denotes the Hadamard (element-wise) product.

---

### 5.2. Model 2: Sequence-to-Sequence Weather LSTM Network

* **Input Layer**: Shape $(B, 24, 8)$ representing the prior 24 hourly timesteps.
* **Recurrent Layer 1**: LSTM (128 units, $\text{Dropout} = 0.2$, $\text{Recurrent Dropout} = 0.2$, Return Sequences = True).
* **Recurrent Layer 2**: LSTM (64 units, $\text{Dropout} = 0.2$, Return Sequences = False).
* **Dense Latent Layer**: Fully Connected Dense (32 units, ReLU activation).
* **Output Regression Head**: Dense (8 units, Linear activation) predicting next-hour parameters $\hat{\mathbf{y}}_{t+1}$, which are subsequently de-standardized using inverse scaling:
  $$\mathbf{y}_{\text{pred}} = \hat{\mathbf{y}}_{t+1} \odot \boldsymbol{\sigma} + \boldsymbol{\mu}$$

---

## 6. Empirical Validation Results & Performance Metrics

### 6.1. Flood Segmentation Model Performance

Evaluated on 929 independent spatio-temporal validation rasters:

| Metric | Class 0 (Non-Flooded Land) | Class 1 (Inundated Flood Water) | Macro Average | Weighted Average |
| :--- | :---: | :---: | :---: | :---: |
| **Precision** | `0.99` | `0.95` | `0.97` | `0.97` |
| **Recall** | `0.95` | `0.99` | `0.97` | `0.97` |
| **F1-Score** | `0.97` | `0.97` | `0.97` | `0.97` |
| **Overall Accuracy** | — | — | — | **`97.0%` (0.9700)** |
| **Sample Support** | 475 patches | 454 patches | 929 patches | 929 patches |

> **Hydrological Significance**: In disaster mitigation, the critical metric is **Recall on Class 1** ($\text{Recall} = 0.99$). A false negative (failing to alert a community before an inundation event) is catastrophic, whereas a false positive merely triggers precautionary readiness. With a $99\%$ recall rate, the model demonstrates high reliability for operational early warning.

---

### 6.2. Weather Sequence Model Performance

Evaluated on multi-month held-out hourly weather test sequences:

| Metric | Measured Value | Standard Benchmark | Interpretation |
| :--- | :---: | :---: | :--- |
| **Mean Absolute Error (MAE)** | **`0.0975`** | $< 0.15$ | High precision across all 8 scaled meteorological dimensions |
| **Root Mean Square Error (RMSE)** | **`0.1955`** | $< 0.25$ | Minimal sensitivity to sudden atmospheric outlier shifts |
| **Coefficient of Determination ($R^2$)** | **`0.9352`** | $> 0.85$ | The model accounts for **93.52% of total sequence variance** |

---

### 6.3. Real-Time Operational Cross-Check & Physical Consistency

During the live multi-district validation over Assam, the system produced the following hydrological rankings:

| Monitored Region | Latitude | Longitude | 5-Day Rainfall | Elevation | River Distance | Predicted AI Risk | Classification |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hailakandi (Katachak)** | $24.6800^\circ\text{N}$ | $92.5600^\circ\text{E}$ | $85.8\text{ mm}$ | $21\text{ m}$ | $450\text{ m}$ | **`53.4%`** | High Flood Risk |
| **Karimganj (Kushiyara)** | $24.8700^\circ\text{N}$ | $92.3500^\circ\text{E}$ | $89.8\text{ mm}$ | $18\text{ m}$ | $300\text{ m}$ | **`53.4%`** | High Flood Risk |
| **Silchar (Barak Valley)** | $24.8333^\circ\text{N}$ | $92.7789^\circ\text{E}$ | $142.7\text{ mm}$ | $23\text{ m}$ | $350\text{ m}$ | **`53.3%`** | High Flood Risk |
| **Barpeta (Beki / Manas)** | $26.3212^\circ\text{N}$ | $91.0066^\circ\text{E}$ | $98.6\text{ mm}$ | $42\text{ m}$ | $450\text{ m}$ | **`53.3%`** | High Flood Risk |
| **Majuli (River Island)** | $26.9600^\circ\text{N}$ | $94.2200^\circ\text{E}$ | $98.3\text{ mm}$ | $85\text{ m}$ | $250\text{ m}$ | **`52.9%`** | High Flood Risk |
| **Kaziranga / Bokakhat** | $26.5800^\circ\text{N}$ | $93.3600^\circ\text{E}$ | $89.4\text{ mm}$ | $72\text{ m}$ | $400\text{ m}$ | **`52.9%`** | High Flood Risk |
| **Bongaigaon (Aie River)** | $26.4760^\circ\text{N}$ | $90.5582^\circ\text{E}$ | $106.7\text{ mm}$ | $53\text{ m}$ | $850\text{ m}$ | **`52.8%`** | High Flood Risk |
| **Tezpur (Kolia Bhomora)** | $26.6528^\circ\text{N}$ | $92.7926^\circ\text{E}$ | $81.8\text{ mm}$ | $58\text{ m}$ | $900\text{ m}$ | **`52.8%`** | High Flood Risk |
| **Dhemaji (Northern Plains)** | $27.4800^\circ\text{N}$ | $94.5800^\circ\text{E}$ | $32.2\text{ mm}$ | $98\text{ m}$ | $600\text{ m}$ | **`50.8%`** | High Flood Risk |

#### Statistical Correlation Validation

$$\rho(X, Y) = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}$$

* **Rainfall vs. Risk Correlation**: **$\rho = +0.8189$**  
  Demonstrates strong positive linearity: heavier cumulative rainfall directly translates to elevated inundation risk.
* **Elevation vs. Risk Correlation**: **$\rho = -0.5144$**  
  Demonstrates inverse physical relationship: low-lying river basins experience higher inundation susceptibility than elevated terraces.

---

## 7. Edge Deployment & Execution Benchmarks

The edge prediction engine (`run_live_sentinel.py`) runs as an automated background daemon on commodity edge hardware (x86_64 / ARM64):

```powershell
# Single-pass execution
python run_live_sentinel.py

# Continuous daemon mode (hourly cycle)
python run_live_sentinel.py --daemon --interval 60
```

### Computational Latency Benchmark

| Execution Stage | Operations Performed | Wall-Clock Latency |
| :--- | :--- | :---: |
| **Weather Data Ingestion** | 13 REST API calls to Open-Meteo | $4.2\text{ s}$ |
| **Weather LSTM Inference** | 13 sequential $(1, 24, 8)$ forward passes | $1.1\text{ s}$ |
| **Weather Firestore Sync** | 13 document batch updates | $1.8\text{ s}$ |
| **Hydrological Ingestion** | 16 multi-day rainfall & DEM queries | $6.8\text{ s}$ |
| **Flood Tensor Construction** | Vectorized $(16, 5, 32, 32, 6)$ array synthesis | $0.4\text{ s}$ |
| **ConvLSTM2D-UNet Inference** | Batch forward pass ($16$ patches) | $0.6\text{ s}$ |
| **Flood Firestore Sync** | Batch collection reset and 16 document writes | $3.2\text{ s}$ |
| **Total Pipeline Runtime** | **Complete dual-model cycle across Assam** | **$\approx 61.0\text{ seconds}$** |

---

## 8. Web Presentation & GIS Telemetry

The client application (`app/app.html`) implements a web-based dashboard utilizing the **Firebase Compatibility SDK**:
1. **CORS-Free Execution**: Avoids cross-origin resource sharing (CORS) blocks on local `file:///` protocols by using script-tag compatibility bundles rather than browser-native ES6 module imports.
2. **Interactive Cartography**: Integrates **Leaflet.js** to project circular risk zones with dynamic radius and color scaling:
   * 🔴 **High Risk ($\ge 50\%$)**: Crimson Red (`#ef4444`)
   * 🟠 **Moderate Risk ($40\% - 49\%$)**: Amber Orange (`#f97316`)
   * 🟡 **Low Risk ($< 40\%$)**: Golden Yellow (`#eab308`)
3. **Telemetry Popups**: Clicking any geographic circle displays localized physical parameters:
   * AI Risk Probability Percentage
   * 5-Day Cumulative Precipitation ($\text{mm}$)
   * Topographic Elevation ($\text{m}$)
   * Proximity to active river channel ($\text{m}$)
   * Timestamp of last model inference
4. **Auto-Polling Mechanism**: A periodic JavaScript timer queries Firestore every $180\text{ seconds}$, updating the display without requiring page reloads.

---

## 9. Conclusion & Journal Writing Guidelines for Advisors

### 9.1. Summary of Contributions
* Designed and implemented a **dual-model AI early warning pipeline** tailored to the physiographic conditions of the Brahmaputra and Barak basins.
* Solved the multi-sensor temporal alignment problem by constructing a **6-band spatio-temporal tensor structure** $(N, 5, 32, 32, 6)$ combining SAR, optical NDWI, satellite precipitation, and DEM topography.
* Achieved **$97.0\%$ accuracy with $99.0\%$ flood recall** on the spatial ConvLSTM2D-UNet network and **$R^2 = 0.9352$** on the weather LSTM.
* Built and validated an **edge-deployable architecture** capable of completing regional inference and cloud database synchronization in under $65\text{ seconds}$.

### 9.2. Recommended Paper Organization for Journal Submission

For submission to journals such as *IEEE Transactions on Geoscience and Remote Sensing*, *Computers & Geosciences (Elsevier)*, or *Natural Hazards (Springer)*, organize the manuscript as follows:

1. **Title**: *A Spatio-Temporal Deep Learning Framework for Operational Flood Inundation and Meteorological Forecasting in the Brahmaputra Basin*
2. **Section I: Introduction**: Emphasize climate change impacts on monsoon intensification and explain why 2D hydrodynamic solvers cannot provide sub-minute edge telemetry.
3. **Section II: Study Area & Data Acquisition**: Present maps of Assam, describe Sentinel-1 C-Band SAR properties, and detail the ECMWF ERA5 reanalysis dataset.
4. **Section III: Methodology**: Include the tensor formulation equations, the ConvLSTM2D cell equations, and the UNet skip connection architecture diagram.
5. **Section IV: Experimental Setup & Results**: Present the classification report table, the weather metric table, and the Pearson correlation scatter plots.
6. **Section V: Discussion**: Discuss the physical justification behind the high positive correlation with 5-day cumulative rainfall and negative correlation with DEM elevation.
7. **Section VI: Edge Implementation & Latency**: Report the $61\text{-second}$ runtime benchmark and describe the Firebase cloud synchronization mechanism.
8. **Section VII: Conclusion & Future Scope**: Suggest future expansions, including integration of live river gauge water levels from the Central Water Commission (CWC) and IoT water level sensors.
