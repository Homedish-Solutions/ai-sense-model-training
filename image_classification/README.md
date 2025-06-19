# 📦 AI Fruit Image Classifier with MediaPipe Model Maker
# Image classification using Transfer Learning

This project is a **fruit image classification model** built using [MediaPipe Model Maker](https://developers.google.com/mediapipe/solutions/vision/image_classifier), powered by TensorFlow 2. It loads a dataset of fruit images, trains a custom classifier, and exports the model in `.tflite` format for on-device usage (e.g., in mobile apps).

---

## 🧠 Features

- Train an image classification model using MediaPipe and TensorFlow 2
- Modular project structure with reusable utilities
- Automatically unzips and loads dataset
- Visualizes sample images per class (optional)
- Evaluates and exports trained model to `exported_model/model.tflite`

---

## 🗂️ Project Structure

```
project/
│
├── main.py                  # Main script for training and evaluation
├── dataset/                 # Contains FruitsDataset.zip (input) and extracted data
├── exported_model/          # Folder where the trained model will be exported
└── utils/                   # Utility functions split by responsibility
    ├── __init__.py          # Exposes functions from preprocessor and exporter
    ├── preprocessor.py      # Handles unzipping, label extraction, and image previews
    └── exporter.py          # Exports trained model
```

---

## ⚙️ Setup Instructions

### 1. 📦 Install dependencies

```bash
pip install tensorflow==2.15.0
pip install mediapipe-model-maker
```

> Make sure you're using Python 3.8–3.10 (compatible with TensorFlow 2.15).

### 2. 🗂️ Prepare dataset

- Put your zipped dataset at: `dataset/FruitsDataset.zip`
- Your ZIP must follow this folder structure:

```
FruitsDataset/
├── Apple/
│   ├── image1.jpg
│   └── ...
├── Banana/
│   └── ...
└── ...
```

### 3. ▶️ Run training

```bash
python main.py
```

### 4. ✅ Output

- After training, your model will be saved to:
```
exported_model/model.tflite
```

---

## 📸 Optional: Show Sample Images

To visualize example images for each class, uncomment this line in `main.py`:

```python
utils.show_examples(image_path, labels, num_examples=5)
```

---

## 🛠️ Customization

- **Change epochs**: In `main.py`, update:
```python
hparams = image_classifier.HParams(export_dir="exported_model", epochs=50)
```

- **Model type**: Change `EFFICIENTNET_LITE2` to any supported model:
```python
image_classifier.SupportedModels.EFFICIENTNET_LITE0
```

---

## 🧪 Example Use Cases

- Fruit quality detection on mobile
- Educational apps for kids to recognize fruits
- Embedded vision applications using EdgeTPU-compatible models

---

## 📄 License

This project is for educational and personal use only. Please check MediaPipe and TensorFlow licenses before commercial use.