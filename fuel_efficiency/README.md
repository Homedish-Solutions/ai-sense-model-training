# Fuel Efficiency Prediction Model

This folder contains a simple Linear Regression model to predict vehicle fuel efficiency (MPG) based on engine and car specs. The model is trained using the Auto MPG dataset.

## 📊 Dataset

- Source: [UCI Machine Learning Repository - Auto MPG Dataset](https://www.kaggle.com/datasets/uciml/autompg-dataset)
Features used:
- Cylinders
- Displacement
- Horsepower
- Weight
- Acceleration
- Model Year
- Origin (USA, Europe, Japan - one-hot encoded)

Target variable:
- MPG (Miles Per Gallon)


### 🧠 Algorithm

* **Type**: Regression  
* **Model**: Deep Neural Network (DNN) using TensorFlow/Keras  
* **Layers**:
  * 2 hidden layers with 64 neurons each and ReLU activation  
  * 1 output layer (linear) for predicting MPG (Miles Per Gallon)
* **Normalization**: Standardization using Z-score:  
  \[(x - mean) / std\]  
  Ensures all features are on a similar scale for better convergence
* **Loss Function**: Mean Squared Error (MSE)  
* **Optimizer**: RMSprop (learning rate = 0.001)

---

### 📊 Evaluation

* **Evaluation Metrics**:
  * MAE (Mean Absolute Error)
  * MSE (Mean Squared Error)
* **Data Split**:
  * 80% training data
  * 20% test data
* **Model Performance**:
  * `loss`, `mae`, and `mse` printed using `.evaluate()`
  * Test predictions compared to ground truth labels

* **TFLite File**:  
`automobile.tflite` — Optimized model ready for mobile apps

---

### 📂 File Details

* `fuel_efficiency_predictor.py`: Python script with end-to-end training pipeline
* `data/auto-mpg.csv`: UCI Auto MPG dataset
* `automobile.keras`: Saved model in Keras format
* `automobile.tflite`: Model ready for TensorFlow Lite usage

# Model Training

## Data Normalization

* Normalization scales all feature values to a **common scale** so the model can learn more effectively.
* Formula:

  $$
  \text{normalized } x = \frac{x - \text{mean}}{\text{std}}
  $$

  This is called **Z-score normalization** or **standardization**.
* Normalization brings feature values roughly within a similar range (centered near 0 with unit variance).
* Benefits:

  * Neural networks converge faster and more reliably(Neural networks learn faster and more reliably when input features are on a similar scale.).
  * Avoids one feature dominating others due to larger range.

---

## Activation Functions

* Activation functions introduce **non-linearity** so the network can learn complex patterns beyond simple linear relationships.
* Example:

  $$
  \text{ReLU}(x) = \max(0, x)
  $$
* **ReLU (Rectified Linear Unit):**

  * Widely used because it is simple and efficient.
  * Allows positive values to pass unchanged, zeros out negatives.
  * Helps **reduce the vanishing gradient problem** (common with sigmoid/tanh).
* **Vanishing Gradient Problem:**

  * Happens when gradients shrink too much during backpropagation, slowing or stopping learning in early layers.
  * Sigmoid and tanh functions saturate for large inputs, causing small gradients.
  * ReLU avoids this by having a constant gradient for positive inputs.
* Other common activations:

  * **Sigmoid:** outputs between 0 and 0.25, can suffer from vanishing gradients.
  * **Tanh:** outputs between -1 and 1, also can suffer from vanishing gradients.
  * **Softmax:** used in classification to produce probability distributions.

---

## Model Architecture

* In this project:

  * Two hidden layers, each with 64 neurons and ReLU activation.
  * One output layer with 1 neuron (no activation) for regression output.

---

## Optimizers

* Purpose:
  To **update the neural network weights** to minimize the loss function.
* How it works:

  * After a prediction, calculate the error (loss).
  * The optimizer uses this loss to adjust weights to improve predictions next time.
* Example scenario:

  * Model predicts 40 MPG instead of true 35 MPG.
  * Loss function measures error severity (e.g., MSE).
  * Optimizer adjusts weights so future predictions get closer to 35.
* Common optimizers:

  * **SGD (Stochastic Gradient Descent):** simple, good for small datasets.
  * **Momentum:** accelerates learning by accumulating gradients.
  * **RMSprop:** adapts learning rate per parameter, avoids zig-zagging.
  * **Adam:** combines Momentum and RMSprop, generally works well across tasks.

---

## Loss Functions

* Measure how far off predictions are from true values.
* **Mean Squared Error (MSE):**

  $$
  \text{MSE} = \text{mean}((y_{\text{pred}} - y_{\text{true}})^2)
  $$

  * Penalizes larger errors more (due to squaring).
  * Suitable for regression where large mistakes are costly.
* **Mean Absolute Error (MAE):**

  $$
  \text{MAE} = \text{mean}(|y_{\text{pred}} - y_{\text{true}}|)
  $$

  * Treats all errors equally.
  * More robust to outliers.
* In this project:

  * MSE is used as the **loss function** (what the model tries to minimize).
  * MAE is used as an additional **performance metric** for evaluation.

---

## Choosing Layer Size (e.g., 64 neurons)

* 64 neurons is a common choice balancing:

  * Model capacity (enough to learn complex patterns, but not too large to overfit).
  * Training time (larger layers take longer).
  * Overfitting risk (too many neurons may memorize training data).
* Other sizes like 32, 128, or 256 neurons are often experimented with.
* Larger layers increase complexity but also increase risk of overfitting, especially with limited data.
* To find the best size, use **validation loss** or **cross-validation** to evaluate performance on unseen data.

---

## Hyperparameter Tuning Using Validation Loss and Cross-Validation

### Validation Loss

* Dataset is split into:

  * **Training data:** used to train the model.
  * **Validation data:** used to evaluate the model during training without updating weights.
* The model does not learn from validation data, so validation loss indicates how well the model generalizes to unseen data.
* Validation loss measures the error the model makes on the validation dataset.
* Example validation losses for different neuron counts:

| Number of Neurons | Validation Loss |
| ----------------- | --------------- |
| 32                | 8.4             |
| 64                | 5.7 (best)      |
| 128               | 6.2             |

* Choose 64 neurons since it results in the lowest validation loss.

### Cross-Validation (More Reliable)

* Instead of a single train/validation split, cross-validation repeats the process multiple times with different splits:

  * Data is split into *K* folds (e.g., 5).
  * Train on *K-1* folds, validate on the remaining fold.
  * Repeat *K* times, each fold used once for validation.
  * Average validation loss across all splits provides a robust estimate.
* Helps reduce bias from any single random data split.

### Example using `KFold` from `sklearn`:

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5)
for train_index, val_index in kf.split(data):
    train_data, val_data = data[train_index], data[val_index]
    # Train model and compute validation loss here
```

