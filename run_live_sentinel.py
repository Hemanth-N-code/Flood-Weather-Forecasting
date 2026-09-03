"""
Sentinel EdgeAI: Real-Time Dual-Mode Prediction Engine
=======================================================
Runs genuine real-time inference using trained neural models:
1. LSTM Sequence Model for next-hour Weather Forecasting.
2. ConvLSTM2D-UNet Hybrid Model for spatial Flood Risk Forecasting.
Synchronizes live results to Firebase Firestore.
"""

import sys
import os
import time
import argparse
from datetime import datetime, timezone, timedelta
IST = timezone(timedelta(hours=5, minutes=30))

# Add edge directory to Python module search path
EDGE_DIR = os.path.join(os.path.dirname(__file__), "edge")
if EDGE_DIR not in sys.path:
    sys.path.insert(0, EDGE_DIR)

from weather_edge_prediction import run_weather_prediction
from flood_edge_prediction import run_flood_prediction

def print_banner():
    banner = r"""
  ╔═══════════════════════════════════════════════════════════════════╗
  ║    🛰️  ASSAM FLOOD & WEATHER AI SENTINEL - LIVE PREDICTOR        ║
  ║         EdgeAI Autonomous Neural Monitoring Engine                ║
  ╚═══════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def execute_pipeline():
    """Runs one full pass of live weather and flood inference."""
    start_time = time.time()
    current_time = datetime.now(IST).strftime("%Y-%m-%d %I:%M:%S %p IST")
    print(f"\n⏱️  Pipeline Triggered at: {current_time}")
    print("=" * 70)

    # 1. Weather Inference
    print("\n[STEP 1/2] Executing Weather LSTM Inference across Assam...")
    try:
        weather_results = run_weather_prediction()
    except Exception as e:
        print(f"❌ Weather prediction encountered an error: {e}")
        weather_results = {}

    # 2. Flood Inference
    print("\n[STEP 2/2] Executing ConvLSTM2D-UNet Flood Risk Inference...")
    try:
        flood_results = run_flood_prediction()
    except Exception as e:
        print(f"❌ Flood prediction encountered an error: {e}")
        flood_results = []

    elapsed = time.time() - start_time
    print("=" * 70)
    print(f"✨ Sentinel Pipeline finished in {elapsed:.1f} seconds.")
    print(f"📊 Summary: {len(weather_results)} weather stations & {len(flood_results)} flood zones updated in Firestore.")
    print("=" * 70 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Assam Live Flood & Weather AI Sentinel Engine")
    parser.add_argument("--daemon", action="store_true", help="Run continuously in background mode")
    parser.add_argument("--interval", type=int, default=60, help="Interval in minutes between prediction cycles (default: 60)")
    args = parser.parse_args()

    print_banner()

    if args.daemon:
        print(f"🔄 Daemon mode activated. Updating every {args.interval} minutes.")
        print("   Press Ctrl+C to terminate the daemon.\n")
        try:
            while True:
                execute_pipeline()
                print(f"⏳ Sleeping for {args.interval} minutes until next cycle...")
                time.sleep(args.interval * 60)
        except KeyboardInterrupt:
            print("\n🛑 Daemon stopped by user.")
    else:
        execute_pipeline()

if __name__ == "__main__":
    main()
