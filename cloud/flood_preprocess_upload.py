import numpy as np
import rasterio
import glob
import os
import firebase_admin
from firebase_admin import credentials, storage
from google.colab import drive
drive.mount('/content/drive')
# ===============================
# 1. SETUP & FIREBASE
# ===============================
if firebase_admin._apps:
    firebase_admin.delete_app(firebase_admin.get_app())

# Load credentials dynamically from service account JSON or environment variable
CRED_PATH = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "your_service_account.json")
if not os.path.exists(CRED_PATH):
    alt_path = os.path.join(os.path.dirname(__file__), "..", "edge", "your_service_account.json")
    if os.path.exists(alt_path):
        CRED_PATH = alt_path

if os.path.exists(CRED_PATH):
    cred = credentials.Certificate(CRED_PATH)
else:
    raise FileNotFoundError("Firebase credentials not found. Please provide your_service_account.json or set GOOGLE_APPLICATION_CREDENTIALS.")

firebase_admin.initialize_app(cred, {
    'storageBucket': 'flood-weather-app.firebasestorage.app'
})

# ===============================
# 2. DYNAMIC FILE SELECTION (5 Most Recent)
# ===============================
DATA_DIR = "/content/drive/MyDrive/flood_prediction_latest"
PATCH_SIZE = 32
STRIDE = 32
SEQ_LEN = 5

# Find all TIFs and sort them by date (Newest first)
all_tifs = sorted(glob.glob(f"{DATA_DIR}/flood_slice_*.tif"), reverse=True)

if len(all_tifs) < SEQ_LEN:
    raise ValueError(f"❌ Found only {len(all_tifs)} files. Need {SEQ_LEN} days of data.")

# Select the 5 most recent and sort chronologically (Oldest -> Newest)
selected_files = all_tifs[:SEQ_LEN]
selected_files.reverse()

print("📂 Sequence identified for LSTM window:")
for f in selected_files:
    print(f"  -> {os.path.basename(f)}")

# ===============================
# 3. PROCESSING & SLICING (Padding Fix)
# ===============================
def normalize_and_pad(img):
    img = img.astype(np.float32)
    img = np.nan_to_num(img)
    
    # Check band count
    num_bands = img.shape[0]
    
    # If only 5 bands exist, add a 6th band filled with zeros
    if num_bands == 5:
        # Create a zero-filled band with same Height and Width
        padding = np.zeros((1, img.shape[1], img.shape[2]), dtype=np.float32)
        img = np.concatenate([img, padding], axis=0)
    
    # Now that we are GUARANTEED 6 bands, apply normalization
    img[2] /= 50.0   # Precipitation
    img[3] /= 3000.0 # Elevation
    img[4] /= 45.0   # Slope
    img[5] /= 5000.0 # Distance (Will be 0 if padded)
        
    return img

imgs = []
transform = None

for f in selected_files:
    with rasterio.open(f) as src:
        print(f"Processing: {os.path.basename(f)} | Found Bands: {src.count}")
        img = src.read()
        transform = src.transform
        # Normalize AND fix shape inconsistencies
        processed_img = normalize_and_pad(img)
        imgs.append(np.transpose(processed_img, (1, 2, 0)))

h, w, _ = imgs[0].shape
X, locations = [], []

for i in range(0, h - PATCH_SIZE, STRIDE):
    for j in range(0, w - PATCH_SIZE, STRIDE):
        # Every sequence is now guaranteed to be (5, 32, 32, 6)
        seq = [imgs[t][i:i+PATCH_SIZE, j:j+PATCH_SIZE] for t in range(SEQ_LEN)]
        X.append(seq)
        
        lon, lat = rasterio.transform.xy(transform, i + 16, j + 16)
        locations.append((lat, lon))

# Conversion to Numpy will now succeed!
X = np.array(X, dtype=np.float32)
locations = np.array(locations, dtype=np.float32)

print(f"✅ Homogeneous Array Created! Final Shape: {X.shape}")

# ===============================
# 4. SAVE & UPLOAD
# ===============================
LOCAL_SAVE = "latest_forecast_data.npz"
np.savez_compressed(LOCAL_SAVE, X=np.array(X, dtype=np.float32), locations=np.array(locations, dtype=np.float32))

bucket = storage.bucket()
blob = bucket.blob("flood_preprocess/latest_forecast_data.npz")
blob.upload_from_filename(LOCAL_SAVE)
blob.make_public()

print(f"🚀 SUCCESS! Data updated. Public URL: {blob.public_url}")