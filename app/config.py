import os

# Path where your trained model is saved
MODEL_PATH = os.path.join("saved_models", "image_classification_model.h5")

# Image size used during training
IMG_WIDTH = 224
IMG_HEIGHT = 224

# If you have class names (same order as training dataset)
CLASS_NAMES = [
    "paper",
    "cardboard",
    "plastic",
    "organic",
    "glass",
    # add the rest...
]
