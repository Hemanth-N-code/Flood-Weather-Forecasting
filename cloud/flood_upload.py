# ===============================
# 1. INSTALL & IMPORTS
# ===============================
!pip install firebase-admin

import firebase_admin
from firebase_admin import credentials, storage
import os

# ===============================
# 2. FIREBASE CONFIGURATION
# ===============================
# Force delete the old Firebase session so we can load the new bucket settings
if firebase_admin._apps:
    print("🧹 Clearing old Firebase session...")
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

print("🔄 Initializing Firebase with Storage Bucket...")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'flood-weather-app.firebasestorage.app'
})

# ===============================
# 3. UPLOAD LOGIC
# ===============================
CHECKPOINT_PATH = "/content/drive/MyDrive/hemanth/flood_model_checkpoint.keras"

def upload_model_to_firebase(local_file_path, remote_file_name):
    print(f"\n☁️ Preparing to upload {remote_file_name} to Firebase Storage...")
    
    if not os.path.exists(local_file_path):
        print(f"❌ ERROR: Could not find the file at {local_file_path}")
        print("Please check your Google Drive path!")
        return

    try:
        bucket = storage.bucket()
        blob = bucket.blob(f"models/{remote_file_name}")
        blob.upload_from_filename(local_file_path)
        blob.make_public()
        
        print("✅ Upload Successful!")
        print(f"🔗 Public Download URL for your Raspberry Pi:")
        print(blob.public_url)
        
    except Exception as e:
        print(f"❌ Upload Failed: {e}")

upload_model_to_firebase(CHECKPOINT_PATH, "flood_model_v1.keras")