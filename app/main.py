from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import io

from .model import load_model
from .config import IMG_WIDTH, IMG_HEIGHT, CLASS_NAMES

app = FastAPI()
model = None


@app.get("/")
def home():
    return {"message": "Waste Classifier API is running 🚀"}

@app.on_event("startup")
async def startup_event():
    global model
    model = load_model()

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    """Receives an image file, preprocesses it, and returns the predicted class."""
    contents = await file.read()

    # Open and preprocess image
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Run prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = CLASS_NAMES[predicted_class_index] if CLASS_NAMES else str(predicted_class_index)

    return {
        "predicted_class_index": int(predicted_class_index),
        "predicted_class_name": predicted_class_name
    }
