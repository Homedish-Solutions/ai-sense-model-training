# House Price Prediction Model



## 📊 Dataset

- Source: [UCI Machine Learning Repository - California House Price Dataset](https://www.kaggle.com/datasets/shibumohapatra/house-price)
longitude – Geographic coordinate (east-west)

latitude – Geographic coordinate (north-south)

housing_median_age – Median age of houses in the block

total_rooms – Total number of rooms in the block

total_bedrooms – Total number of bedrooms in the block

population – Total population in the block

households – Total number of households in the block

median_income – Median income of households (scaled)

ocean_proximity – Proximity to ocean (e.g., '<1H OCEAN', 'INLAND') — typically one-hot encoded

median_house_value – Target variable representing the median house value

Target variable:
- median_house_value


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
`house_prediction.tflite` — Optimized model ready for mobile apps

---

### 📂 File Details

* `house_prediction.py`: Python script with end-to-end training pipeline
* `data/housing.csv`: UCI Auto MPG dataset
* `house_prediction.keras`: Saved model in Keras format
* `house_prediction.tflite`: Model ready for TensorFlow Lite usage

