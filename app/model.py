import tensorflow as tf
from .config import MODEL_PATH

model = None

def load_model():
    """Load the trained model into memory."""
    global model
    if model is None:
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    return model
