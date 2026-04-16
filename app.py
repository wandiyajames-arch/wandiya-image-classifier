import os
import torch
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, render_template, request, jsonify
from torchvision import transforms

# Import your IntelCNN class and the TF builder
from models.model_def import IntelCNN  
from models.model_def2 import build_wandiya_model_tf

app = Flask(__name__)

CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

# ============================================
# Load Models
# ============================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load PyTorch Model (Using IntelCNN from model_def.py)
try:
    pytorch_model = IntelCNN(num_classes=6).to(device)
    # Adjust path if your file name is different (e.g., 'models/model.pth')
    pytorch_model.load_state_dict(
        torch.load("models/wandiya_model.pth", map_location=device)
    )
    pytorch_model.eval()
    print("✓ PyTorch IntelCNN loaded successfully")
except Exception as e:
    print(f"✗ Error loading PyTorch model: {e}")
    pytorch_model = None

# Load TensorFlow Model
try:
    tf_model = build_wandiya_model_tf()
    tf_model.load_weights("models/wandiya_model.keras")
    print("✓ TensorFlow model loaded successfully")
except Exception as e:
    print(f"✗ Error loading TensorFlow model: {e}")
    tf_model = None

# ============================================
# Preprocessing
# ============================================
pytorch_transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def preprocess_pytorch(image):
    img = Image.open(image).convert("RGB")
    tensor = pytorch_transform(img).unsqueeze(0).to(device)
    return tensor

def preprocess_tensorflow(image):
    img = Image.open(image).convert("RGB").resize((150, 150))
    arr = np.array(img) / 255.0
    # Added Normalization to match the PyTorch logic
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    arr = (arr - mean) / std
    return np.expand_dims(arr, axis=0).astype(np.float32)

# ============================================
# Routes
# ============================================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files["image"]
    model_type = request.form.get("model", "pytorch")

    try:
        if model_type == "pytorch":
            if pytorch_model is None:
                return jsonify({"error": "PyTorch model not loaded"}), 500
            
            tensor = preprocess_pytorch(image)
            with torch.no_grad():
                output = pytorch_model(tensor)
                # Apply softmax to get probabilities
                probs = torch.softmax(output, dim=1)[0]
                idx = probs.argmax().item()
                conf = probs[idx].item() * 100
        else:
            if tf_model is None:
                return jsonify({"error": "TensorFlow model not loaded"}), 500
            
            arr = preprocess_tensorflow(image)
            preds_logits = tf_model.predict(arr, verbose=0)[0]
            # If your TF model doesn't have a Softmax layer at the end:
            preds = tf.nn.softmax(preds_logits).numpy()
            idx = np.argmax(preds)
            conf = float(preds[idx] * 100)

        return jsonify({
            "class": CLASS_NAMES[idx],
            "confidence": f"{conf:.2f}%",
            "model": model_type
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 Starting Wandiya Image Classifier")
    print("="*50)
    print(f"📍 Open http://localhost:5000 in your browser")
    print("Press CTRL+C to stop")
    print("="*50 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)