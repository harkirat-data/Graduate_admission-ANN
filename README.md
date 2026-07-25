# Graduate Admission Chance Predictor

A regression model built with TensorFlow/Keras to predict a student's probability of graduate admission based on standardized test scores and academic profile. Built as a learning project.

## Overview

- **Dataset:** [Graduate Admissions](https://www.kaggle.com/datasets/mohansacharya/graduate-admissions) (Admission_Predict_Ver1.1.csv, 500 rows)
- **Task:** Regression — predict `Chance of Admit` (0–1) from applicant features
- **Framework:** TensorFlow / Keras
- **Features used:** GRE Score, TOEFL Score, University Rating, SOP, LOR, CGPA, Research

## Architecture

```
Dense(7, activation='relu', input_dim=7)
Dense(1, activation='linear')
```

Total params: 64 (all trainable)

## Training

- Optimizer: Adam
- Loss: mean_squared_error
- Epochs: 250, validation_split = 0.2
- Features scaled to [0, 1] with `MinMaxScaler`

## Results

R² score: **0.809**

## Usage

```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

df = pd.read_csv("Admission_Predict_Ver1.1.csv")
df = df.drop("Serial No.", axis=1)

X = df.iloc[:, 0:-1]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

model = keras.models.load_model("model.h5")  # if saved
prediction = model.predict(X_scaled)
```

