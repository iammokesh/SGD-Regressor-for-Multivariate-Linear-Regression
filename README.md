# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Data Preparation Load the dataset using pandas, clean column names, and separate input features (Size, Bedrooms) and target variables (Price, Occupants). Step 2: Data Scaling Apply feature scaling using StandardScaler to normalize the input data for better performance of the SGD model.
Step 3: Model Training Create two SGDRegressor models and train them using the scaled features to predict Price and Occupants.
Step 4: Prediction Take user input, scale it using the same scaler, and use the trained models to predict house price and number of occupants.

## Program:
```python
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by:MOKESH C
RegisterNumber:212225240088  
*/
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
data = fetch_california_housing()
X = data.data[:, :3]
Y=np.column_stack((data.target,data.data[:, 6]))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_train=scaler_X.fit_transform(X_train)
X_test=scaler_X.transform(X_test)
Y_train=scaler_Y.fit_transform(Y_train)
Y_test=scaler_Y.transform(Y_test)
sgd=SGDRegressor(max_iter=1000, tol=1e-3)
multi_output_sgd=MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train,Y_train)
Y_pred=multi_output_sgd.predict(X_test)
Y_pred=scaler_Y.inverse_transform(Y_pred)
Y_test=scaler_Y.inverse_transform(Y_test)
print("\nPredictions:\n",Y_pred[:5])
```

## Output:
<img width="379" height="180" alt="Screenshot 2026-05-16 155353" src="https://github.com/user-attachments/assets/3fc0e83f-6710-4535-b431-a73e5295e204" />




## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
