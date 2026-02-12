import os
import tensorflow as tf
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.regularizers import l1_l2
import cv2
import numpy as np

MODEL_DIR = 'model'
WEIGHTS_DIR = 'weights'
# Check both local weights dir and external_repo/weights if possible, 
# but for now let's stick to a standard path or allow configuration.
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, 'modeldense1.h5')

def ensure_directories():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    if not os.path.exists(WEIGHTS_DIR):
        os.makedirs(WEIGHTS_DIR)

# Removed automatic download to rely on local files as requested

def create_model():
    """Recreates the DenseNet201 architecture used in the external repo"""
    # Note: The external repo used 'pooling=max' and 'weights=imagenet'
    conv_base = DenseNet201(input_shape=(224, 224, 3), include_top=False, pooling='max', weights='imagenet')
    
    model = Sequential()
    model.add(conv_base)
    model.add(BatchNormalization())
    model.add(Dense(2048, activation='relu', kernel_regularizer=l1_l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dense(8, activation='softmax'))
    
    # Freeze layers as per original code (partial freeze)
    train_layers = [layer for layer in conv_base.layers[::-1][:5]]
    for layer in conv_base.layers:
        if layer not in train_layers:
            layer.trainable = False
            
    return model

def load_cancer_model():
    """Loads the model and weights"""
    ensure_directories()
    # download_weights_if_needed() - Removed as per user request
    
    try:
        # Strategy: Create structure then load weights
        # The original repo did model.save('model/model.h5') which saves architecture+weights
        # But also had a separate weights download. 
        # We will try to build the model structure and load the downloaded weights.
        
        # If the full model file exists, we could try loading that, 
        # but creating fresh and loading weights is often more robust across TF versions.
        if os.path.exists(WEIGHTS_PATH):
            try:
                print(f"Attempting to load full model from {WEIGHTS_PATH}...")
                model = tf.keras.models.load_model(WEIGHTS_PATH)
                print("Model loaded successfully using load_model.")
                return model
            except Exception as e:
                print(f"load_model failed ({e}), falling back to create_model + load_weights...")
        
        # Fallback: Create structure then load weights
        model = create_model()
        
        # Compile needed for prediction? Not strictly, but good practice if we were training
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001, decay=0.0001),
                     metrics=["accuracy"],
                     loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1))
                     
        if os.path.exists(WEIGHTS_PATH):
            model.load_weights(WEIGHTS_PATH)
            print("Model weights loaded successfully.")
        else:
            print("Warning: Weights file not found.")
            
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess_image(image_file):
    """
    Preprocesses the image for the model.
    Expects a file-like object or path.
    """
    try:
        # Reset file pointer to beginning
        image_file.seek(0)
        
        # Read image using cv2/numpy
        file_bytes = np.frombuffer(image_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            print("Error: Failed to decode image.")
            return None

        # Convert BGR to RGB (OpenCV loads BGR, Model expects RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Filter (kernel from original repo)
        kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
        img = cv2.filter2D(img, -1, kernel)
        
        # Resize to 224x224
        img = cv2.resize(img, (224, 224))
        
        # Normalize
        img = img / 255.0
        
        # Reshape for model input (batch size 1)
        img_reshape = img.reshape(-1, 224, 224, 3)
        
        return img_reshape
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

CLASS_NAMES = [
    'Benign with Density=1',
    'Malignant with Density=1',
    'Benign with Density=2',
    'Malignant with Density=2',
    'Benign with Density=3',
    'Malignant with Density=3',
    'Benign with Density=4',
    'Malignant with Density=4'
]

def get_prediction_result(predictions):
    """
    Parses the prediction array into a meaningful result.
    Returns: (Diagnosis, Confidence, Density, FullResultDict)
    """
    try:
        # Handle NaNs or invalid outputs from untrained models
        if np.any(np.isnan(predictions)):
            # If model returns NaNs, something is wrong with weights or input
            return "Error", 0.0, "Unknown", {}
            
        pred_idx = np.argmax(predictions)
        confidence = float(predictions[pred_idx])
        
        # If confidence is extremely low, it might be an issue, but let's trust the model
        # The user wants high accuracy, so we should rely on the trained weights.
        
        result_str = CLASS_NAMES[pred_idx]
        
        # Parse result string "Type with Density=N"
        # Example: "Benign with Density=1"
        if ' with ' in result_str:
            parts = result_str.split(' with ')
            diagnosis = parts[0] # Benign or Malignant
            
            # parts[1] is "Density=1"
            # We want just "1", "2", "3", "4"
            if 'Density=' in parts[1]:
                density_val = parts[1].split('Density=')[1]
                density = f"{density_val}"
            else:
                density = parts[1]
        else:
            diagnosis = result_str
            density = "Unknown"
            
        full_results = {CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))}
        
        return diagnosis, confidence, density, full_results
    except Exception as e:
        print(f"Error parsing prediction: {e}")
        return "Error", 0.0, "Unknown", {}
