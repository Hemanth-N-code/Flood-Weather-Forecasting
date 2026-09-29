"""
Academic Diagram Generator for IEEE Publication
================================================
Generates publication-quality, 300-DPI vector-raster architecture diagrams:
1. doc/system_architecture.png: End-to-End Multi-Tier EdgeAI Sentinel Framework
2. doc/convlstm_unet_architecture.png: ConvLSTM2D-UNet Hybrid Neural Architecture
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ArrowStyle

DOC_DIR = os.path.join(os.path.dirname(__file__), "doc")
os.makedirs(DOC_DIR, exist_ok=True)

# Common styling configuration
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 0.8

def draw_box(ax, x, y, w, h, title, subtitle="", box_color="#ffffff", border_color="#64748b", text_color="#1e293b", corner_radius=0.03):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.01,rounding_size={corner_radius}",
        facecolor=box_color,
        edgecolor=border_color,
        linewidth=1.5,
        zorder=2
    )
    ax.add_patch(box)
    
    # Title
    ax.text(
        x + w / 2, y + h * 0.65 if subtitle else y + h * 0.5,
        title,
        ha='center', va='center',
        fontsize=9.5, fontweight='bold',
        color=text_color, zorder=3
    )
    # Subtitle
    if subtitle:
        ax.text(
            x + w / 2, y + h * 0.28,
            subtitle,
            ha='center', va='center',
            fontsize=7.5, fontweight='normal',
            color="#475569", zorder=3
        )

def draw_arrow(ax, x1, y1, x2, y2, color="#475569", width=1.5, label=""):
    arrow = patches.FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='-|>,head_length=5,head_width=3',
        color=color,
        linewidth=width,
        zorder=1
    )
    ax.add_patch(arrow)
    if label:
        ax.text(
            (x1 + x2) / 2, (y1 + y2) / 2 + 0.02,
            label,
            ha='center', va='bottom',
            fontsize=7, fontweight='bold', color=color, zorder=4
        )

# =============================================================================
# 1. GENERATE SYSTEM ARCHITECTURE DIAGRAM
# =============================================================================
def generate_system_architecture():
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.axis('off')
    
    # Background Canvas
    fig.patch.set_facecolor('#f8fafc')
    ax.set_facecolor('#f8fafc')
    
    # Title Header
    ax.text(0.5, 0.98, "EdgeAI Sentinel: End-to-End System Architecture", 
            ha='center', va='top', fontsize=16, fontweight='black', color='#0f172a')
    ax.text(0.5, 0.94, "Multi-Modal Ingestion • Autonomous Edge Inference • Serverless Cloud Telemetry • Public GIS Radar", 
            ha='center', va='top', fontsize=9.5, color='#475569')

    # 5 Major Tiers (Vertical Columns)
    tiers = [
        {"x": 0.01, "w": 0.17, "title": "TIER 1: MULTI-MODAL DATA", "color": "#eff6ff", "border": "#3b82f6"},
        {"x": 0.21, "w": 0.17, "title": "TIER 2: TENSOR FORMULATION", "color": "#f0fdf4", "border": "#22c55e"},
        {"x": 0.41, "w": 0.20, "title": "TIER 3: EDGE AI INFERENCE", "color": "#faf5ff", "border": "#a855f7"},
        {"x": 0.64, "w": 0.17, "title": "TIER 4: CLOUD SYNC", "color": "#fff7ed", "border": "#f97316"},
        {"x": 0.84, "w": 0.15, "title": "TIER 5: GIS DELIVERY", "color": "#fef2f2", "border": "#ef4444"},
    ]
    
    # Draw Tier Containers
    for t in tiers:
        bg_box = FancyBboxPatch((t["x"], 0.05), t["w"], 0.84,
                                boxstyle="round,pad=0.01,rounding_size=0.02",
                                facecolor=t["color"], edgecolor=t["border"], linewidth=1.2, linestyle='--', alpha=0.5)
        ax.add_patch(bg_box)
        ax.text(t["x"] + t["w"]/2, 0.86, t["title"], ha='center', va='center', fontsize=8.5, fontweight='bold', color=t["border"])

    # --- TIER 1: DATA SOURCES ---
    draw_box(ax, 0.025, 0.73, 0.14, 0.09, "Sentinel-1 SAR", "C-Band VV Water Masks (10m)", "#ffffff", "#3b82f6")
    draw_box(ax, 0.025, 0.60, 0.14, 0.09, "Sentinel-2 MSI", "Optical NDWI Moisture (10m)", "#ffffff", "#3b82f6")
    draw_box(ax, 0.025, 0.47, 0.14, 0.09, "NASA GPM IMERG", "5-Day Precipitation History", "#ffffff", "#3b82f6")
    draw_box(ax, 0.025, 0.34, 0.14, 0.09, "NASA SRTM DEM", "30m Elevation & Slope", "#ffffff", "#3b82f6")
    draw_box(ax, 0.025, 0.21, 0.14, 0.09, "JRC Surface Water", "Euclidean River Proximity", "#ffffff", "#3b82f6")
    draw_box(ax, 0.025, 0.08, 0.14, 0.09, "Open-Meteo REST API", "Live 24h Hourly Meteorology", "#ffffff", "#3b82f6")

    # --- TIER 2: TENSORS ---
    draw_box(ax, 0.225, 0.50, 0.14, 0.32, "Spatio-Temporal Tensor\nX ∈ ℝ^(N×5×32×32×6)", "Band 0: SAR Flood Mask\nBand 1: Optical NDWI [0,1]\nBand 2: GPM Rain / 50.0\nBand 3: DEM Height / 3000m\nBand 4: Terrain Slope / 45°\nBand 5: River Dist / 5000m", "#ffffff", "#22c55e", corner_radius=0.02)
    draw_box(ax, 0.225, 0.10, 0.14, 0.32, "Atmospheric Sequence\nW ∈ ℝ^(B×24×8)", "Temp, Dewpoint, MSL Press,\nSurface Press, Rain, Wind,\n+ Diurnal Cyclical Encodings:\nsin(2πh/24) & cos(2πh/24)\nStandardScaler Normalized", "#ffffff", "#22c55e", corner_radius=0.02)

    # --- TIER 3: EDGE AI INFERENCE ---
    draw_box(ax, 0.425, 0.50, 0.17, 0.32, "ConvLSTM2D-UNet Model\n(downloaded_model.keras)", "ConvLSTM2D(32, 3x3) -> BN\nConvLSTM2D(16, 3x3) -> BN\nUNet Bottleneck Conv2D(64)\nUpSampling2D -> Concatenate\nConv2D(1, Sigmoid)\nLatency: ~19.1 ms/patch", "#ffffff", "#a855f7", corner_radius=0.02)
    draw_box(ax, 0.425, 0.10, 0.17, 0.32, "Stacked Weather LSTM\n(weather_lstm_model.h5)", "LSTM(128, Dropout=0.2)\nLSTM(64, Recurrent=0.2)\nDense(32, ReLU) -> Dense(8)\nForecast Horizon: t + 1 hour\nLatency: ~68.3 ms/station", "#ffffff", "#a855f7", corner_radius=0.02)

    # --- TIER 4: CLOUD SYNC ---
    draw_box(ax, 0.655, 0.52, 0.14, 0.28, "Firestore Collection:\n'predictions'", "16 Regional Documents\n• Latitude / Longitude\n• Inundation Risk Score (%)\n• Mean / Peak Probabilities\n• 5d Rain, DEM, River Dist\n• Synchronized in ~1.98 s", "#ffffff", "#f97316", corner_radius=0.02)
    draw_box(ax, 0.655, 0.12, 0.14, 0.28, "Firestore Collection:\n'weather_forecasts'", "13 Station Documents\n• 1-Hour Temp (°C)\n• 1-Hour Rain (mm)\n• 1-Hour Wind (m/s)\n• Updated_At Timestamp\n• Synchronized in ~0.35 s", "#ffffff", "#f97316", corner_radius=0.02)

    # --- TIER 5: GIS PRESENTATION ---
    draw_box(ax, 0.855, 0.45, 0.12, 0.35, "Interactive GIS Map\n(Leaflet.js)", "• Dynamic Risk Circles\n  - Red (>=45% High)\n  - Orange (>=35% Mod)\n  - Green (<35% Low)\n• Statewide Auto-Fit\n• Telemetry Popups", "#ffffff", "#ef4444", corner_radius=0.02)
    draw_box(ax, 0.855, 0.08, 0.12, 0.30, "Public Web Hosting\n(Google CDN)", "URL: flood-weather-app.web.app\n• Global Edge CDN\n• Automated SSL (HTTPS)\n• Client Latency: ~0.41 s\n• Zero Server Maintenance", "#ffffff", "#ef4444", corner_radius=0.02)

    # Arrows Connecting Stages
    draw_arrow(ax, 0.165, 0.65, 0.225, 0.65, "#22c55e", 1.8)
    draw_arrow(ax, 0.165, 0.25, 0.225, 0.25, "#22c55e", 1.8)
    
    draw_arrow(ax, 0.365, 0.65, 0.425, 0.65, "#a855f7", 1.8, "Tensors")
    draw_arrow(ax, 0.365, 0.25, 0.425, 0.25, "#a855f7", 1.8, "Sequences")

    draw_arrow(ax, 0.595, 0.65, 0.655, 0.65, "#f97316", 1.8, "Risk Maps")
    draw_arrow(ax, 0.595, 0.25, 0.655, 0.25, "#f97316", 1.8, "Forecasts")

    draw_arrow(ax, 0.795, 0.65, 0.855, 0.65, "#ef4444", 1.8)
    draw_arrow(ax, 0.795, 0.25, 0.855, 0.25, "#ef4444", 1.8)

    plt.tight_layout()
    out_path = os.path.join(DOC_DIR, "system_architecture.png")
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Generated {out_path} (300 DPI)")

# =============================================================================
# 2. GENERATE CONVLSTM2D-UNET ARCHITECTURE DIAGRAM
# =============================================================================
def generate_convlstm_unet_architecture():
    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=300)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.axis('off')
    
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')

    # Title
    ax.text(0.5, 0.98, "Hybrid ConvLSTM2D-UNet Inundation Segmentation Network", 
            ha='center', va='top', fontsize=16, fontweight='black', color='#0f172a')
    ax.text(0.5, 0.94, "Spatio-Temporal Sequence Learning Coupled with Multi-Scale Skip Decoder", 
            ha='center', va='top', fontsize=10, color='#475569')

    # 1. Input Tensor Block
    draw_box(ax, 0.02, 0.38, 0.13, 0.35, "Input Tensor\n(B, 5, 32, 32, 6)", 
             "• Timesteps: 5 days\n• Spatial: 32×32\n• 6 Physical Channels:\n  c0: SAR Water Mask\n  c1: Optical NDWI\n  c2: Precipitation\n  c3: DEM Elevation\n  c4: Terrain Slope\n  c5: River Proximity", 
             "#f1f5f9", "#0284c7", "#0369a1")

    # 2. ConvLSTM2D Encoder Block
    draw_box(ax, 0.19, 0.58, 0.16, 0.20, "ConvLSTM2D Layer 1\n(32 Filters, 3×3)", "Return Sequences = True\nBatch Normalization + LeakyReLU\nOutput: (B, 5, 32, 32, 32)", "#fdf4ff", "#c026d3", "#86198f")
    draw_box(ax, 0.19, 0.25, 0.16, 0.20, "ConvLSTM2D Layer 2\n(16 Filters, 3×3)", "Return Sequences = False\nBatch Normalization + LeakyReLU\nOutput: (B, 32, 32, 16)", "#fdf4ff", "#c026d3", "#86198f")

    # 3. UNet Encoder & Bottleneck
    draw_box(ax, 0.40, 0.58, 0.15, 0.20, "UNet Encoder Block\nConv2D(32, 3×3)", "LeakyReLU Activation\nSkip Feature Cache\nOutput: (B, 32, 32, 32)", "#ecfdf5", "#059669", "#065f46")
    draw_box(ax, 0.40, 0.25, 0.15, 0.20, "Downsampling Block\nMaxPool2D(2×2)", "Spatial Compression\nOutput: (B, 16, 16, 32)", "#ecfdf5", "#059669", "#065f46")
    draw_box(ax, 0.59, 0.25, 0.15, 0.20, "Bottleneck Block\nConv2D(64, 3×3)", "Deep Representation\nOutput: (B, 16, 16, 64)", "#fef3c7", "#d97706", "#92400e")

    # 4. UNet Decoder & Skip Fusion
    draw_box(ax, 0.59, 0.58, 0.15, 0.20, "Decoder Block\nUpSampling2D(2×2)", "Spatial Re-Expansion\nOutput: (B, 32, 32, 64)", "#eff6ff", "#2563eb", "#1e40af")
    draw_box(ax, 0.78, 0.58, 0.18, 0.20, "Feature Fusion Block\nConcatenate + Conv2D(32)", "Skip Connection (32) + Decoder (64)\nOutput: (B, 32, 32, 32)", "#eff6ff", "#2563eb", "#1e40af")

    # 5. Output Head
    draw_box(ax, 0.78, 0.22, 0.18, 0.24, "Output Head\nConv2D(1, 1x1, Sigmoid)", "Spatial Flood Probability Map\nOutput: (B, 32, 32, 1) in [0.0, 1.0]\n\nProduction Risk Tiers:\n- >=45.0%: High Flood Risk\n- >=35.0%: Moderate Risk\n- <35.0%: Low Flood Risk", "#fef2f2", "#dc2626", "#991b1b")

    # Flow Arrows
    draw_arrow(ax, 0.15, 0.55, 0.19, 0.68, "#86198f", 2)
    draw_arrow(ax, 0.27, 0.58, 0.27, 0.45, "#86198f", 2, "Temporal Sequence")
    draw_arrow(ax, 0.35, 0.35, 0.40, 0.65, "#059669", 2, "Latent State")
    draw_arrow(ax, 0.475, 0.58, 0.475, 0.45, "#059669", 2)
    draw_arrow(ax, 0.55, 0.35, 0.59, 0.35, "#d97706", 2)
    draw_arrow(ax, 0.665, 0.45, 0.665, 0.58, "#2563eb", 2)
    draw_arrow(ax, 0.74, 0.68, 0.78, 0.68, "#2563eb", 2)
    draw_arrow(ax, 0.87, 0.58, 0.87, 0.46, "#dc2626", 2)

    # UNet Skip Connection (Dashed green line across the top)
    skip_arrow = patches.FancyArrowPatch(
        (0.55, 0.72), (0.78, 0.72),
        arrowstyle='-|>,head_length=6,head_width=4',
        color='#059669', linewidth=2.2, linestyle='--', zorder=5
    )
    ax.add_patch(skip_arrow)
    ax.text(0.665, 0.74, "UNet Skip Connection (Feature Preservation)", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#059669')

    # Performance Footnote
    ax.text(0.5, 0.05, "Quantitative Test Validation: Overall Accuracy: 97.00% | Flood Inundation Recall: 99.00% | F1-Score: 0.970 | Single Patch Latency: 19.1 ms", 
            ha='center', va='center', fontsize=9, fontweight='bold', color='#0f172a',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#f1f5f9', edgecolor='#cbd5e1'))

    plt.tight_layout()
    out_path = os.path.join(DOC_DIR, "convlstm_unet_architecture.png")
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Generated {out_path} (300 DPI)")

if __name__ == "__main__":
    generate_system_architecture()
    generate_convlstm_unet_architecture()
