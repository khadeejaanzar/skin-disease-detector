from flask import Flask, render_template, request, jsonify
from PIL import Image
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)

# Load trained model
MODEL_PATH = "skin_disease_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# IMPORTANT: Keep this order exactly the same as during training
CLASSES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]

IMG_SIZE = 224


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Check whether an image was uploaded
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No image selected"}), 400

    try:
        # Open uploaded image
        image = Image.open(file).convert("RGB")

        # Resize exactly like our Kaggle preprocessing
        image = image.resize((IMG_SIZE, IMG_SIZE))

        # Convert to NumPy array
        image_array = np.array(image).astype("float32") / 255.0

        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)

        # Model prediction
        predictions = model.predict(image_array, verbose=0)[0]

        # Find highest probability
        predicted_index = int(np.argmax(predictions))
        predicted_class = CLASSES[predicted_index]
        confidence = float(predictions[predicted_index] * 100)

        # All probabilities
        probabilities = {
            cls: round(float(prob) * 100, 2)
            for cls, prob in zip(CLASSES, predictions)
        }

        return jsonify({
            "prediction": predicted_class,
            "confidence": round(confidence, 2),
            "probabilities": probabilities
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
