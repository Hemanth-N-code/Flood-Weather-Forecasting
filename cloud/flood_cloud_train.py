# ===============================
# 0. MOUNT GOOGLE DRIVE
# ===============================
#from google.colab import drive
#drive.mount('/content/drive')

# ===============================
# 1. INSTALL & IMPORTS
# ===============================
!pip install rasterio

import numpy as np
import rasterio
import glob
import os
import tensorflow as tf
from tensorflow.keras import layers, Model

# ===============================
# 2. CONFIGURATION
# ===============================
DATA_DIR = "/content/drive/MyDrive/hemanth/earthengine_flood_final"
SAVE_PATH = "/content/drive/MyDrive/hemanth/flood_hybrid_final.keras"
CHECKPOINT_PATH = "/content/drive/MyDrive/hemanth/flood_model_checkpoint.keras"

PATCH_SIZE = 32
SEQ_LEN = 5      
MAX_PATCHES = 10 
BATCH_SIZE = 8   

# Reduced to 5 to meet your project deadline
EPOCHS = 5

# Set to 1 because you successfully finished Epoch 1!
INITIAL_EPOCH = 1 

# ===============================
# 3. DATA PROCESSING (Fixes NoneType Error)
# ===============================
def normalize(img):
    img = img.astype(np.float32)
    # Ensure indices match your TIF bands: [Flood, NDWI, Rain, Elev, Slope, Dist]
    img[1] = (img[1] + 1) / 2  
    img[2] /= 50.0             
    img[3] /= 3000.0           
    img[4] /= 45.0             
    img[5] /= 5000.0           
    return img

def load_sequence(idx, file_list):
    """
    Returns empty arrays of the correct shape instead of None 
    to prevent TensorFlow Graph execution errors.
    """
    idx = int(idx)
    file_list = [f.decode("utf-8") if isinstance(f, bytes) else f for f in file_list]
    
    # Define "Empty" returns for error handling
    empty_X = np.zeros((0, SEQ_LEN, PATCH_SIZE, PATCH_SIZE, 6), dtype=np.float32)
    empty_y = np.zeros((0, PATCH_SIZE, PATCH_SIZE, 1), dtype=np.float32)

    seq_imgs = []
    try:
        for k in range(SEQ_LEN + 1):
            path = file_list[idx + k]
            with rasterio.open(path) as src:
                img = src.read()
                img = np.nan_to_num(img)
                img = normalize(img)
                img = np.transpose(img, (1, 2, 0))
                seq_imgs.append(img)
    except Exception as e:
        # Silently skip bad files, the .filter() in the pipeline handles the empty return
        return empty_X, empty_y

    X_patches, y_patches = [], []
    h, w, _ = seq_imgs[0].shape

    for _ in range(MAX_PATCHES * 2): 
        if len(X_patches) >= MAX_PATCHES: break
        
        i = np.random.randint(0, h - PATCH_SIZE)
        j = np.random.randint(0, w - PATCH_SIZE)

        seq = [seq_imgs[t][i:i+PATCH_SIZE, j:j+PATCH_SIZE] for t in range(SEQ_LEN)]
        target = seq_imgs[SEQ_LEN][i:i+PATCH_SIZE, j:j+PATCH_SIZE, 0:1]

        # Priority sampling for flood pixels so the model actually learns!
        flood_ratio = np.mean(target)
        if flood_ratio > 0.01 or np.random.rand() < 0.1:
            X_patches.append(seq)
            y_patches.append(target)

    if not X_patches: 
        return empty_X, empty_y
        
    return np.array(X_patches, dtype=np.float32), np.array(y_patches, dtype=np.float32)

# ===============================
# 4. TF DATASET PIPELINE
# ===============================
def tf_wrapper(idx, file_list):
    X, y = tf.numpy_function(load_sequence, [idx, file_list], [tf.float32, tf.float32])
    # Set shapes explicitly so the model knows what to expect
    X.set_shape((None, SEQ_LEN, PATCH_SIZE, PATCH_SIZE, 6))
    y.set_shape((None, PATCH_SIZE, PATCH_SIZE, 1))
    return X, y

def get_dataset(files):
    idxs = list(range(len(files) - SEQ_LEN - 1))
    ds = tf.data.Dataset.from_tensor_slices(idxs)
    ds = ds.shuffle(len(idxs))
    # num_parallel_calls is key for speed
    ds = ds.map(lambda i: tf_wrapper(i, files), num_parallel_calls=tf.data.AUTOTUNE)
    # Filter out the empty arrays returned by failed loads
    ds = ds.filter(lambda x, y: tf.shape(x)[0] > 0)
    ds = ds.unbatch().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

# ===============================
# 5. MODEL ARCHITECTURE
# ===============================
def build_hybrid_model():
    inputs = layers.Input(shape=(SEQ_LEN, PATCH_SIZE, PATCH_SIZE, 6))
    x = layers.ConvLSTM2D(32, (3,3), padding="same", return_sequences=True)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(16, (3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    
    # Spatial Head (U-Net style skip)
    c1 = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    p1 = layers.MaxPooling2D()(c1)
    b = layers.Conv2D(64, 3, activation="relu", padding="same")(p1)
    u1 = layers.UpSampling2D()(b)
    
    concat = layers.concatenate([u1, c1])
    c2 = layers.Conv2D(32, 3, activation="relu", padding="same")(concat)
    outputs = layers.Conv2D(1, 1, activation="sigmoid")(c2)
    return Model(inputs, outputs)

# ===============================
# 6. RUN LOGIC
# ===============================
all_files = sorted(glob.glob(f"{DATA_DIR}/*.tif"))
if not all_files:
    print("❌ No files found! Check your DATA_DIR path.")
else:
    split = int(0.8 * len(all_files))
    train_ds = get_dataset(all_files[:split])
    val_ds = get_dataset(all_files[split:])

    if os.path.exists(CHECKPOINT_PATH):
        print(f"🔄 Resuming from checkpoint, starting at Epoch {INITIAL_EPOCH + 1}...")
        model = tf.keras.models.load_model(CHECKPOINT_PATH)
    else:
        print("🆕 Starting fresh...")
        model = build_hybrid_model()
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), 
                      loss='binary_crossentropy', 
                      metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=CHECKPOINT_PATH,
        save_best_only=False, 
        verbose=1
    )

    print("🚀 Training...")
    model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=EPOCHS, 
        initial_epoch=INITIAL_EPOCH, 
        steps_per_epoch=50,       # <--- SPEED FIX: Limits an epoch to 50 batches
        validation_steps=10,      # <--- SPEED FIX: Speeds up the validation phase
        callbacks=[cp_callback]
    )
    
    model.save(SAVE_PATH)
    print(f"✅ Done! Final Model saved to {SAVE_PATH}")