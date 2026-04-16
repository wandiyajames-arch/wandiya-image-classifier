import io
import torch
import torchvision.transforms as T
from flask import Flask, request, jsonify
from PIL import Image
from models.model_def import IntelCNN  # Ensure model_def.py is in the same folder

app = Flask(__name__)

# --- 1. Load the Model ---
num_classes = 6
model = IntelCNN(num_classes=num_classes)
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
model.eval()

# --- 2. Class Labels ---
# Update this list to match the order of your folders in Colab
classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# --- 3. Image Preprocessing ---
def transform_image(image_bytes):
    transform = T.Compose([
        T.Resize((150, 150)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

# --- 4. The Prediction Route ---
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    img_bytes = file.read()
    
    try:
        # Preprocess
        tensor = transform_image(img_bytes)
        
        # Predict
        with torch.no_grad():
            outputs = model(tensor)
            _, predicted = torch.max(outputs, 1)
            class_idx = predicted.item()
            
        return jsonify({
            'class_index': class_idx,
            'class_name': classes[class_idx]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)