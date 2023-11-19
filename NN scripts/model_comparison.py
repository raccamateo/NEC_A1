# Import necessary libraries
from my_neural_network import MyNeuralNetwork
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd  # If you're using pandas for data handling

import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset from the provided URL
url = "https://raw.githubusercontent.com/raccamateo/NEC_BP_LR/main/normalized_wine_data.csv"
wine_data = pd.read_csv(url)
print(wine_data.head())  # Just print the first few rows to check


X = wine_data.drop('quality', axis=1)
y = wine_data['quality']

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize and train the scikit-learn MLPRegressor
mlp = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=200, random_state=42)
mlp.fit(X_train, y_train)
mlp_predictions = mlp.predict(X_val)
mlp_mse = mean_squared_error(y_val, mlp_predictions)

# Initialize and train the scikit-learn LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
linear_reg_predictions = linear_reg.predict(X_val)
linear_reg_mse = mean_squared_error(y_val, linear_reg_predictions)

# Initialize and train your custom MyNeuralNetwork
my_nn = MyNeuralNetwork(num_layers=3, units_per_layer=[X_train.shape[1], 100, 1], num_epochs=100, learning_rate=0.01, momentum=0.9, validation_split=0.2)
my_nn.fit(X_train, y_train)
my_nn_predictions = my_nn.predict(X_val)
my_nn_mse = mean_squared_error(y_val, my_nn_predictions)

# Print comparison results
print(f"Mean Squared Error (MLPRegressor): {mlp_mse}")
print(f"Mean Squared Error (LinearRegression): {linear_reg_mse}")
print(f"Mean Squared Error (MyNeuralNetwork): {my_nn_mse}")
