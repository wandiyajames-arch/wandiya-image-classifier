# 🏔️ Intel Image Classifier

A deep learning web application that classifies natural scene images into **6 categories** — powered by a CNN built with **PyTorch** or **TensorFlow**, served via a **Flask** REST API.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0%2B-black?logo=flask)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11%2B-EE4C2C?logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21%2B-FF6F00?logo=tensorflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

This project is based on the [Intel Image Classification dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification), which contains ~25,000 images of natural scenes from around the world.

**Scene categories:**

| Label | Description |
|-------|-------------|
| `buildings` | Urban structures and architecture |
| `forest` | Dense woodland and tree cover |
| `glacier` | Ice fields and frozen landscapes |
| `mountain` | Peaks, ridges, and highland terrain |
| `sea` | Ocean, lakes, and coastal views |
| `street` | Roads, pavements, and city scenes |

---

## 🗂️ Project Structure

```
intel-image-classifier/
├── app/
│   ├── __init__.py
│   ├── routes.py            # Flask API endpoints
│   ├── predict.py           # Inference logic (PyTorch + TensorFlow)
│   └── utils.py             # Image preprocessing helpers
├── models/
│   ├── pytorch_model.pth    # Trained PyTorch weights
│   └── tf_model/            # Saved TensorFlow model directory
├── notebooks/
│   └── training.ipynb       # Model training & evaluation notebook
├── static/
│   └── uploads/             # Temp storage for uploaded images
├── templates/
│   └── index.html           # (Optional) web UI
├── requirements.txt
├── config.py
├── run.py                   # Entry point (development)
└── wsgi.py                  # Gunicorn entry point (production)
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.10 or higher
- `pip` package manager
- (Optional) CUDA-enabled GPU for faster training

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/intel-image-classifier.git
cd intel-image-classifier
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `tensorflow` and `tensorflow-cpu` are both listed — install only the one that matches your hardware. Remove the other from `requirements.txt` before installing to avoid conflicts.

```bash
# GPU machine
pip install tensorflow>=2.21.0

# CPU-only machine
pip install tensorflow-cpu>=2.21.0
```

---

## 🚀 Running the App

### Development server

```bash
python run.py
```

The app will start at `http://localhost:5000`.

### Production server (Gunicorn)

```bash
gunicorn --workers 4 --bind 0.0.0.0:8000 wsgi:app
```

---

## 🔌 API Reference

### `POST /predict`

Upload an image file and receive a scene classification.

**Request**

```
Content-Type: multipart/form-data
```

| Field | Type | Description |
|-------|------|-------------|
| `image` | `file` | JPEG or PNG image to classify |
| `backend` | `string` | `"pytorch"` or `"tensorflow"` (optional, default: `"pytorch"`) |

**Response**

```json
{
  "category": "mountain",
  "confidence": {
    "buildings": 0.02,
    "forest": 0.05,
    "glacier": 0.11,
    "mountain": 0.74,
    "sea": 0.05,
    "street": 0.03
  },
  "inference_time_ms": 48
}
```

**Error response**

```json
{
  "error": "Unsupported file type. Please upload a JPEG or PNG image."
}
```

### `GET /health`

Returns API health status.

```json
{ "status": "ok", "model": "pytorch" }
```

---

## 🧠 Model Architecture

Both backends use a **CNN** trained on the Intel dataset with the following design:

```
Input (150×150×3)
  └── Conv2D (32) → BatchNorm → ReLU → MaxPool
  └── Conv2D (64) → BatchNorm → ReLU → MaxPool
  └── Conv2D (128) → BatchNorm → ReLU → MaxPool
  └── Flatten
  └── Dense (256) → Dropout (0.5)
  └── Dense (6) → Softmax
```

Transfer learning from **ResNet-50** (pretrained on ImageNet) is also supported as an alternative backbone — see `config.py` to switch.

---

## 📊 Training

Open the Jupyter notebook to train from scratch:

```bash
jupyter notebook notebooks/training.ipynb
```

**Training configuration (defaults):**

| Parameter | Value |
|-----------|-------|
| Image size | 150 × 150 |
| Batch size | 32 |
| Epochs | 25 |
| Optimizer | Adam (lr = 1e-3) |
| Loss | Categorical Cross-Entropy |

**Visualising results** — the notebook uses `matplotlib` to plot training/validation accuracy and the confusion matrix.

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `flask` | ≥ 3.0.0 | Web framework & REST API |
| `Werkzeug` | ≥ 3.0.0 | WSGI utilities (used by Flask) |
| `torch` | ≥ 2.11.0 | PyTorch deep learning backend |
| `torchvision` | ≥ 0.26.0 | Image transforms & pretrained models |
| `tensorflow` / `tensorflow-cpu` | ≥ 2.21.0 | TensorFlow deep learning backend |
| `Pillow` | ≥ 12.2.0 | Image loading & preprocessing |
| `numpy` | ≥ 2.4.4 | Numerical operations |
| `matplotlib` | ≥ 3.8.0 | Training visualisation |
| `gunicorn` | ≥ 21.2.0 | Production WSGI server |

---

## 🗃️ Dataset

Download the **Intel Image Classification** dataset from Kaggle:

```
https://www.kaggle.com/datasets/puneet6060/intel-image-classification
```

Place the extracted data at:

```
data/
├── train/
│   ├── buildings/
│   ├── forest/
│   ├── glacier/
│   ├── mountain/
│   ├── sea/
│   └── street/
├── test/
└── val/
```

---

## 📝 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Intel Image Classification dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) by Puneet Bansal on Kaggle
- [PyTorch](https://pytorch.org/) and [TensorFlow](https://www.tensorflow.org/) open-source teams
- [Flask](https://flask.palletsprojects.com/) micro-framework

---

*Built by [Wandiya](https://github.com/wandiya) · GAAI-AIMS Programme*
