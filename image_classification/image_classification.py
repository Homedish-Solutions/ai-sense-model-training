import os
import tensorflow as tf
assert tf.__version__.startswith('2')
from mediapipe_model_maker import image_classifier

import utils


zip_file_name = "dataset/FruitsDataset"
zip_path = f"{zip_file_name}.zip"
    
    # Unzip dataset
extract_dir = utils.unzip_dataset(zip_path, extract_to="dataset")

image_path = zip_file_name  # path to unzipped dataset folder
print(f"Dataset path: {image_path}") 

labels = utils.get_labels(image_path)
print(f"Labels found: {labels}")


# utils.show_examples(image_path, labels, num_examples=5)
# Load dataset
data = image_classifier.Dataset.from_folder(image_path)
train_data, remaining_data = data.split(0.9)
test_data, validation_data = remaining_data.split(0.5)

# Set model options
spec = image_classifier.SupportedModels.EFFICIENTNET_LITE2
hparams = image_classifier.HParams(export_dir="exported_model",epochs=50)
options = image_classifier.ImageClassifierOptions(supported_model=spec, hparams=hparams)

 # Create and train model
model = image_classifier.ImageClassifier.create(
    train_data = train_data,
    validation_data = validation_data,
    options=options,
)

# Evaluate model
loss, acc = model.evaluate(test_data)
print(f'Test loss:{loss}, Test accuracy:{acc}')

# Export model
model.export_model()

