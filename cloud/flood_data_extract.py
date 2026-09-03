import ee
import datetime
import time
import os

# ===============================
# INIT
# ===============================
ee.Authenticate()
ee.Initialize(project='shining-rush-471915-s7')

# ===============================
# SETTINGS
# ===============================
# Region: Assam
assam = ee.Geometry.Rectangle([89.5, 24.5, 94, 28])
SCALE = 100
EXPORT_FOLDER = "flood_prediction_latest"
# Full local path to your mounted Google Drive folder
DRIVE_PATH = f"/content/drive/MyDrive/{EXPORT_FOLDER}" 
LOOKBACK_DAYS = 7 

# 🔥 GET TODAY
today = datetime.date.today()

def start_export_task(target_date):
    """Stacks 6 bands and exports only if file doesn't exist."""
    
    # 1. GENERATE FILENAME
    filename = f"flood_slice_{target_date}"
    file_path = os.path.join(DRIVE_PATH, f"{filename}.tif")

    # 2. CHECK IF FILE EXISTS IN DRIVE
    if os.path.exists(file_path):
        print(f"⏩ Skipping {target_date}: File already exists in Drive.")
        return # Exit the function without starting a task

    print(f"📡 {target_date} is missing. Preparing Earth Engine extraction...")

    start = ee.Date(str(target_date))
    end = start.advance(1, 'day')

    # --- DATA LAYERS (Fixed DataType to Float32) ---
    s1 = ee.ImageCollection("COPERNICUS/S1_GRD") \
        .filterBounds(assam) \
        .filterDate(start.advance(-5, 'day'), end) \
        .filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .median().select('VV').float()

    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
        .filterBounds(assam) \
        .filterDate(start.advance(-3, 'day'), start.advance(3, 'day')) \
        .median()
    ndwi = s2.normalizedDifference(['B3', 'B8']).rename("NDWI").float()

    rain = ee.ImageCollection("NASA/GPM_L3/IMERG_V07") \
        .filterBounds(assam) \
        .filterDate(start, end) \
        .select("precipitation").sum().rename("RAIN").float()

    dem = ee.Image("USGS/SRTMGL1_003").rename("ELEVATION").float()
    slope = ee.Terrain.slope(dem).rename("SLOPE").float()
    dist = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence") \
        .gt(50).fastDistanceTransform(100).sqrt().rename("DIST").float()

    # Stack and Align
    combined = ee.Image.cat([
        s1.rename("FLOOD"), 
        ndwi,
        rain,
        dem.reproject('EPSG:4326', None, SCALE).float(),
        slope.reproject('EPSG:4326', None, SCALE).float(),
        dist.reproject('EPSG:4326', None, SCALE).float()
    ]).float() 

    # 3. TRIGGER EXPORT
    task = ee.batch.Export.image.toDrive(
        image=combined.clip(assam),
        description=filename,
        folder=EXPORT_FOLDER,
        fileNamePrefix=filename,
        region=assam,
        scale=SCALE,
        maxPixels=1e13
    )
    task.start()
    print(f"✅ Export task SUBMITTED for: {target_date}")

# ===============================
# LOOP: Only process what's needed
# ===============================
print(f"🔍 Checking Drive for missing slices in the last {LOOKBACK_DAYS} days...")
for i in range(LOOKBACK_DAYS):
    target_date = today - datetime.timedelta(days=i)
    start_export_task(target_date)
    time.sleep(1) # Small delay for Drive I/O stability

print("\n🔥 Synchronization Check Complete.")