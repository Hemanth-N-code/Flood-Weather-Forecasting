"""
Academic Diagram Generator for IEEE Publication (Spacious & Clean Layout)
========================================================================
Generates publication-quality, 300-DPI vector-raster architecture diagrams
with spacious layouts, zero text overlapping, and clear typography:
1. doc/system_architecture.png: End-to-End Multi-Tier EdgeAI Sentinel Framework
2. doc/convlstm_unet_architecture.png: ConvLSTM2D-UNet Hybrid Neural Architecture
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

DOC_DIR = os.path.join(os.path.dirname(__file__), "doc")
os.makedirs(DOC_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 0.8

def draw_card(ax, x, y, w, h, title, body_lines, card_color="#ffffff", border_color="#3b82f6", header_color=None, title_color="#0f172a"):
    """Draws a spacious, professional UI card with a header and bullet points."""
    # Outer card
    card = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        facecolor=card_color,
        edgecolor=border_color,
        linewidth=1.4,
        zorder=2
    )
    ax.add_patch(card)
    
    # Optional header background banner
    if header_color:
        header_h = h * 0.28
        header_patch = FancyBboxPatch(
            (x, y + h - header_h), w, header_h,
            boxstyle="round,pad=0.01,rounding_size=0.015",
            facecolor=header_color,
            edgecolor=border_color,
            linewidth=1.0,
            zorder=3
        )
        ax.add_patch(header_patch)
        title_y = y + h - header_h / 2
    else:
        title_y = y + h - 0.035

    # Title text
    ax.text(
        x + w / 2, title_y,
        title,
        ha='center', va='center',
        fontsize=9.0, fontweight='bold',
        color=title_color, zorder=4
    )
    
    # Body lines (with generous line spacing and vertical centering)
    if body_lines:
        line_count = len(body_lines)
        available_h = (h * 0.70) if header_color else (h - 0.06)
        line_spacing = available_h / (line_count + 1)
        start_y = (y + h * 0.70) if header_color else (y + h - 0.065)
        
        for i, line in enumerate(body_lines):
            curr_y = start_y - (i + 0.6) * line_spacing
            ax.text(
                x + 0.012, curr_y,
                line,
                ha='left', va='center',
                fontsize=7.2, color="#334155",
                zorder=4
            )

def draw_flow_arrow(ax, x1, y1, x2, y2, color="#475569", width=1.5, label=""):
    arrow = patches.FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='-|>,head_length=5,head_width=3.5',
        color=color,
        linewidth=width,
        zorder=1
    )
    ax.add_patch(arrow)
    if label:
        ax.text(
            (x1 + x2) / 2, (y1 + y2) / 2 + 0.018,
            label,
            ha='center', va='bottom',
            fontsize=7.5, fontweight='bold', color=color, zorder=5,
            bbox=dict(boxstyle='round,pad=0.15', facecolor='#ffffff', edgecolor='#e2e8f0', alpha=0.9)
        )

# =============================================================================
# 1. GENERATE SPACIOUS SYSTEM ARCHITECTURE DIAGRAM
# =============================================================================
def generate_system_architecture():
    # Large 20x11 canvas for zero overlap
    fig, ax = plt.subplots(figsize=(20, 11), dpi=300)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.axis('off')
    
    fig.patch.set_facecolor('#f8fafc')
    ax.set_facecolor('#f8fafc')
    
    # Title Banner
    ax.text(0.5, 0.98, "EdgeAI Sentinel: End-to-End System Architecture", 
            ha='center', va='top', fontsize=18, fontweight='black', color='#0f172a')
    ax.text(0.5, 0.945, "Coupled Multi-Modal Remote Sensing • Autonomous Edge Inference • Serverless Cloud Telemetry", 
            ha='center', va='top', fontsize=11, color='#64748b')

    # 5 Big Columns / Tiers
    col_w = 0.165
    gap = 0.035
    start_x = 0.015
    
    tiers = [
        {"name": "TIER 1: DATA SOURCES", "sub": "Multi-Sensor Observation Lake", "border": "#2563eb", "bg": "#eff6ff"},
        {"name": "TIER 2: TENSORS", "sub": "Feature Engineering & Transforms", "border": "#16a34a", "bg": "#f0fdf4"},
        {"name": "TIER 3: EDGE AI INFERENCE", "sub": "Dual-Neural Prediction Engine", "border": "#9333ea", "bg": "#faf5ff"},
        {"name": "TIER 4: CLOUD SYNC", "sub": "Google Cloud Firestore Buffer", "border": "#ea580c", "bg": "#fff7ed"},
        {"name": "TIER 5: GIS DELIVERY", "sub": "Real-Time Web Application", "border": "#dc2626", "bg": "#fef2f2"}
    ]
    
    for i, t in enumerate(tiers):
        cx = start_x + i * (col_w + gap)
        # Background Column Outline
        col_box = FancyBboxPatch((cx, 0.04), col_w, 0.86,
                                boxstyle="round,pad=0.01,rounding_size=0.02",
                                facecolor=t["bg"], edgecolor=t["border"], linewidth=1.2, linestyle='--', alpha=0.45)
        ax.add_patch(col_box)
        
        # Column Title
        ax.text(cx + col_w / 2, 0.875, t["name"], ha='center', va='center', fontsize=9.0, fontweight='black', color=t["border"])
        ax.text(cx + col_w / 2, 0.855, t["sub"], ha='center', va='center', fontsize=7.2, color='#64748b')

    # --- TIER 1 CARDS (x = 0.02) ---
    x1 = start_x + 0.008
    w1 = col_w - 0.016
    draw_card(ax, x1, 0.72, w1, 0.11, "Sentinel-1 SAR Radar", 
              ["• C-Band VV Polarisation (10m)", "• All-weather cloud penetration", "• Binary ground truth water mask"], 
              border_color="#3b82f6", header_color="#dbeafe")

    draw_card(ax, x1, 0.58, w1, 0.11, "Sentinel-2 MSI Optical", 
              ["• Multi-Spectral Bands (10m)", "• NDWI = (Green-NIR)/(Green+NIR)", "• Captures surface water moisture"], 
              border_color="#3b82f6", header_color="#dbeafe")

    draw_card(ax, x1, 0.44, w1, 0.11, "NASA GPM IMERG Daily", 
              ["• Satellite Precipitation History", "• 5-Day Cumulative Rainfall (mm)", "• Antecedent catchment moisture"], 
              border_color="#3b82f6", header_color="#dbeafe")

    draw_card(ax, x1, 0.30, w1, 0.11, "NASA SRTM 30m DEM", 
              ["• Topographic ground height (m)", "• Derived slope gradient (deg)", "• Prevents mountain false alerts"], 
              border_color="#3b82f6", header_color="#dbeafe")

    draw_card(ax, x1, 0.16, w1, 0.11, "JRC Global Surface Water", 
              ["• Long-term historical water map", "• Euclidean distance to rivers (m)", "• Proximity to Brahmaputra/Barak"], 
              border_color="#3b82f6", header_color="#dbeafe")

    draw_card(ax, x1, 0.05, w1, 0.09, "Open-Meteo REST API", 
              ["• Live hourly weather observations", "• 13 Assam meteorological stations"], 
              border_color="#3b82f6", header_color="#dbeafe")

    # --- TIER 2 CARDS (x = 0.215) ---
    x2 = start_x + 1 * (col_w + gap) + 0.008
    w2 = col_w - 0.016
    draw_card(ax, x2, 0.48, w2, 0.35, "Flood Tensor Formulation\nX in R^(N x 5 x 32 x 32 x 6)", 
              ["• N = 16 Assam monitored basins",
               "• T = 5 consecutive days (t-4 to t)",
               "• Spatial = 32 x 32 grid patch",
               "• Channel 0: SAR Flood Mask {0, 1}",
               "• Channel 1: Optical NDWI in [0, 1]",
               "• Channel 2: Rain (mm) / 50.0",
               "• Channel 3: DEM Height / 3000m",
               "• Channel 4: Slope / 45 degrees",
               "• Channel 5: River Dist / 5000m"],
              border_color="#22c55e", header_color="#dcfce7")

    draw_card(ax, x2, 0.08, w2, 0.35, "Weather Matrix Formulation\nW in R^(B x 24 x 8)", 
              ["• B = Batch size (13 stations)",
               "• Sequence = Rolling 24 hours",
               "• 8 Normalized Features:",
               "  - 2m Temperature (deg C)",
               "  - 2m Dewpoint (deg C)",
               "  - Surface & MSL Pressure (hPa)",
               "  - Rainfall & Wind Speed",
               "  - Cyclical: sin(2*pi*h / 24)",
               "  - Cyclical: cos(2*pi*h / 24)",
               "• StandardScaler Normalization"],
              border_color="#22c55e", header_color="#dcfce7")

    # --- TIER 3 CARDS (x = 0.415) ---
    x3 = start_x + 2 * (col_w + gap) + 0.008
    w3 = col_w - 0.016
    draw_card(ax, x3, 0.48, w3, 0.35, "ConvLSTM2D-UNet Model\n(downloaded_model.keras)", 
              ["• Spatio-Temporal Inundation Head",
               "• ConvLSTM2D(32, 3x3) + BN",
               "• ConvLSTM2D(16, 3x3) + BN",
               "• UNet Bottleneck Conv2D(64, 3x3)",
               "• UpSampling2D + Skip Connection",
               "• Output: (16, 32, 32, 1) probability",
               "• Accuracy: 97.0% | Recall: 99.0%",
               "• Inference Latency: ~19.1 ms/patch",
               "• Correlation: +0.819 Rain, -0.514 DEM"],
              border_color="#a855f7", header_color="#f3e8ff")

    draw_card(ax, x3, 0.08, w3, 0.35, "Weather Sequence LSTM\n(weather_lstm_model.h5)", 
              ["• Recurrent Atmospheric Forecaster",
               "• Stacked LSTM Architecture:",
               "  - LSTM Layer 1 (128 units, drop=0.2)",
               "  - LSTM Layer 2 (64 units, drop=0.2)",
               "  - Dense Latent (32 units, ReLU)",
               "  - Dense Head (8 units, Linear)",
               "• Forecast Horizon: t + 1 hour ahead",
               "• R2 = 0.9352 | MAE = 0.0975",
               "• Inference Latency: ~68.3 ms/station"],
              border_color="#a855f7", header_color="#f3e8ff")

    # --- TIER 4 CARDS (x = 0.615) ---
    x4 = start_x + 3 * (col_w + gap) + 0.008
    w4 = col_w - 0.016
    draw_card(ax, x4, 0.48, w4, 0.35, "Cloud Firestore:\n'predictions' Collection", 
              ["• 16 Basin Alert Documents:",
               "  - Place name & Geocoordinates",
               "  - Peak Flood Probability (%)",
               "  - Mean Inundation Score",
               "  - Status: High/Moderate/Low Risk",
               "  - 5-day Rainfall, DEM, River Dist",
               "  - Sync Timestamp (IST)",
               "• Production Thresholds:",
               "  - High Risk: >= 45.00%",
               "  - Moderate Risk: >= 35.00%",
               "• Write Latency: ~1.98 s (Batch)"],
              border_color="#f97316", header_color="#ffedd5")

    draw_card(ax, x4, 0.08, w4, 0.35, "Cloud Firestore:\n'weather_forecasts' Collection", 
              ["• 13 Station Telemetry Documents:",
               "  - Station Name & Coordinates",
               "  - Predicted 1-Hour Temp (deg C)",
               "  - Predicted 1-Hour Rain (mm)",
               "  - Predicted 1-Hour Wind (m/s)",
               "  - Updated_At Timestamp (IST)",
               "• Decoupled Egress Architecture",
               "  (Zero open ports on edge device)",
               "• Write Latency: ~0.35 s (Batch)"],
              border_color="#f97316", header_color="#ffedd5")

    # --- TIER 5 CARDS (x = 0.815) ---
    x5 = start_x + 4 * (col_w + gap) + 0.008
    w5 = col_w - 0.016
    draw_card(ax, x5, 0.48, w5, 0.35, "Interactive GIS Radar\n(Leaflet.js)", 
              ["• Real-Time Hazard Visualization",
               "• Dynamic Risk Marker Buffers:",
               "  - Red (High Inundation Risk)",
               "  - Orange (Moderate Inundation Risk)",
               "  - Yellow (Low Inundation Risk)",
               "• Statewide Auto-Fit Bounds",
               "• 4-Cell Telemetry Popups",
               "• Auto-polling every 180 seconds"],
              border_color="#ef4444", header_color="#fee2e2")

    draw_card(ax, x5, 0.08, w5, 0.35, "Global Web Hosting\n(Firebase CDN)", 
              ["• Public URL:",
               "  flood-weather-app.web.app",
               "• Google Edge Infrastructure:",
               "  - Automated HTTPS / SSL",
               "  - Single-Page App (SPA) rewrite",
               "  - Browser Compat SDK",
               "• CDN Delivery Latency: ~0.41 s",
               "• Zero Server Maintenance Cost"],
              border_color="#ef4444", header_color="#fee2e2")

    # Connecting Flow Arrows (Horizontally aligned, clean offset)
    draw_flow_arrow(ax, x1 + w1, 0.65, x2, 0.65, "#16a34a", 1.8, "Tensors")
    draw_flow_arrow(ax, x1 + w1, 0.25, x2, 0.25, "#16a34a", 1.8, "Sequences")

    draw_flow_arrow(ax, x2 + w2, 0.65, x3, 0.65, "#9333ea", 1.8, "Inference")
    draw_flow_arrow(ax, x2 + w2, 0.25, x3, 0.25, "#9333ea", 1.8, "Inference")

    draw_flow_arrow(ax, x3 + w3, 0.65, x4, 0.65, "#ea580c", 1.8, "Telemetry")
    draw_flow_arrow(ax, x3 + w3, 0.25, x4, 0.25, "#ea580c", 1.8, "Forecasts")

    draw_flow_arrow(ax, x4 + w4, 0.65, x5, 0.65, "#dc2626", 1.8, "Web Sync")
    draw_flow_arrow(ax, x4 + w4, 0.25, x5, 0.25, "#dc2626", 1.8, "CDN Push")

    plt.tight_layout()
    out_path = os.path.join(DOC_DIR, "system_architecture.png")
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Regenerated {out_path} with spacious layout.")

# =============================================================================
# 2. GENERATE SPATIOUS CONVLSTM2D-UNET ARCHITECTURE DIAGRAM
# =============================================================================
def generate_convlstm_unet_architecture():
    # Large 20x11 canvas
    fig, ax = plt.subplots(figsize=(20, 11), dpi=300)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.axis('off')
    
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')

    # Title Banner
    ax.text(0.5, 0.98, "Hybrid ConvLSTM2D-UNet Inundation Segmentation Network", 
            ha='center', va='top', fontsize=18, fontweight='black', color='#0f172a')
    ax.text(0.5, 0.945, "Spatio-Temporal Recurrent Encoding Coupled with Multi-Scale Skip Decoder", 
            ha='center', va='top', fontsize=11, color='#64748b')

    # 1. Input Tensor Card (Leftmost, x = 0.02)
    draw_card(ax, 0.02, 0.28, 0.15, 0.54, "Input Multi-Modal Tensor\n(B, 5, 32, 32, 6)", 
              ["• Timesteps: 5 temporal days",
               "• Spatial Resolution: 32 x 32 grid",
               "• 6 Physical Channels:",
               "  - c0: SAR C-Band Water Mask",
               "  - c1: Sentinel-2 Optical NDWI",
               "  - c2: NASA GPM Rain / 50.0",
               "  - c3: SRTM DEM Elevation / 3000m",
               "  - c4: Terrain Slope / 45 deg",
               "  - c5: Euclidean River Distance / 5000m",
               "• Vectorized Tensor Synthesis",
               "• Normalized Input Range: [0.0, 1.0]"],
              border_color="#0284c7", header_color="#e0f2fe", title_color="#0369a1")

    # 2. ConvLSTM2D Recurrent Layers (Column 2, x = 0.21)
    draw_card(ax, 0.21, 0.58, 0.17, 0.24, "ConvLSTM2D Layer 1\n(32 Filters, 3x3 Kernel)", 
              ["• Return Sequences: True",
               "• Batch Normalization + LeakyReLU",
               "• Captures continuous temporal transitions",
               "• Output: (B, 5, 32, 32, 32)"], 
              border_color="#c026d3", header_color="#fae8ff", title_color="#86198f")

    draw_card(ax, 0.21, 0.26, 0.17, 0.24, "ConvLSTM2D Layer 2\n(16 Filters, 3x3 Kernel)", 
              ["• Return Sequences: False (Last Step)",
               "• Batch Normalization + LeakyReLU",
               "• Condenses temporal history into 2D state",
               "• Output: (B, 32, 32, 16)"], 
              border_color="#c026d3", header_color="#fae8ff", title_color="#86198f")

    # 3. UNet Encoder & Bottleneck (Column 3, x = 0.42)
    draw_card(ax, 0.42, 0.58, 0.17, 0.24, "UNet Encoder Block\nConv2D(32, 3x3 Kernel)", 
              ["• Spatial Feature Extraction",
               "• LeakyReLU Activation",
               "• Feeds Skip Connection Line",
               "• Output: (B, 32, 32, 32)"], 
              border_color="#059669", header_color="#d1fae5", title_color="#065f46")

    draw_card(ax, 0.42, 0.26, 0.17, 0.24, "Downsampling Block\nMaxPool2D(2x2)", 
              ["• Spatial Dimension Reduction",
               "• Compression: 32x32 -> 16x16",
               "• Expands receptive field",
               "• Output: (B, 16, 16, 32)"], 
              border_color="#059669", header_color="#d1fae5", title_color="#065f46")

    # 4. Bottleneck & Decoder (Column 4, x = 0.63)
    draw_card(ax, 0.63, 0.26, 0.16, 0.24, "Bottleneck Block\nConv2D(64, 3x3 Kernel)", 
              ["• Deep Non-Linear Representation",
               "• Channel Expansion: 32 -> 64",
               "• High-level abstract feature learning",
               "• Output: (B, 16, 16, 64)"], 
              border_color="#d97706", header_color="#fef3c7", title_color="#92400e")

    draw_card(ax, 0.63, 0.58, 0.16, 0.24, "Decoder Block\nUpSampling2D(2x2)", 
              ["• Spatial Re-Expansion",
               "• Restores 16x16 -> 32x32",
               "• Bilinear/Nearest interpolation",
               "• Output: (B, 32, 32, 64)"], 
              border_color="#2563eb", header_color="#dbeafe", title_color="#1e40af")

    # 5. Skip Fusion & Output Head (Column 5, x = 0.83)
    draw_card(ax, 0.83, 0.58, 0.16, 0.24, "Feature Fusion Block\nConcatenate + Conv2D(32)", 
              ["• Skip Connection Fusion:",
               "  Encoder (32) + Decoder (64)",
               "• Preserves fine spatial boundaries",
               "• Conv2D(32, 3x3) refinement",
               "• Output: (B, 32, 32, 32)"], 
              border_color="#2563eb", header_color="#dbeafe", title_color="#1e40af")

    draw_card(ax, 0.83, 0.22, 0.16, 0.28, "Output Head\nConv2D(1, 1x1, Sigmoid)", 
              ["• Probability Heatmap: (B, 32, 32, 1)",
               "• Pixel Values: [0.0, 1.0]",
               "• Production Hazard Tiers:",
               "  - High Risk: >= 45.00%",
               "  - Moderate Risk: >= 35.00%",
               "  - Low Risk: < 35.00%",
               "• Peak & Mean Risk Parsing"], 
              border_color="#dc2626", header_color="#fee2e2", title_color="#991b1b")

    # Flow Arrows
    draw_flow_arrow(ax, 0.17, 0.60, 0.21, 0.70, "#86198f", 2)
    draw_flow_arrow(ax, 0.295, 0.58, 0.295, 0.50, "#86198f", 2, "Temporal Sequence")
    draw_flow_arrow(ax, 0.38, 0.38, 0.42, 0.70, "#059669", 2, "Latent State")
    draw_flow_arrow(ax, 0.505, 0.58, 0.505, 0.50, "#059669", 2)
    draw_flow_arrow(ax, 0.59, 0.38, 0.63, 0.38, "#d97706", 2)
    draw_flow_arrow(ax, 0.71, 0.50, 0.71, 0.58, "#2563eb", 2)
    draw_flow_arrow(ax, 0.79, 0.70, 0.83, 0.70, "#2563eb", 2)
    draw_flow_arrow(ax, 0.91, 0.58, 0.91, 0.50, "#dc2626", 2)

    # UNet Skip Connection (Dashed green curve across top with clear clearance)
    skip_arrow = patches.FancyArrowPatch(
        (0.59, 0.83), (0.83, 0.83),
        arrowstyle='-|>,head_length=6,head_width=4',
        color='#059669', linewidth=2.2, linestyle='--', zorder=5
    )
    ax.add_patch(skip_arrow)
    ax.text(0.71, 0.845, "UNet Skip Connection (Feature Preservation)", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#059669')

    # Footnote Box at Bottom
    footnote_text = "Validation Metrics (929 Test Rasters): Overall Accuracy: 97.00% | Flood Recall: 99.00% | F1-Score: 0.970 | Inference Latency: 19.1 ms/patch"
    ax.text(0.5, 0.08, footnote_text, 
            ha='center', va='center', fontsize=9.5, fontweight='bold', color='#0f172a',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8fafc', edgecolor='#94a3b8', linewidth=1.2))

    plt.tight_layout()
    out_path = os.path.join(DOC_DIR, "convlstm_unet_architecture.png")
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Regenerated {out_path} with spacious layout.")

if __name__ == "__main__":
    generate_system_architecture()
    generate_convlstm_unet_architecture()
