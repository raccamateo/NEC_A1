from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


# Initialize the MLPRegressor with desired parameters
mlp_regressor = MLPRegressor(hidden_layer_sizes=(100,),  
                             activation='relu',
                             solver='adam',
                             alpha=0.0001,
                             batch_size='auto',
                             learning_rate='constant',
                             learning_rate_init=0.001,
                             max_iter=200,
                             random_state=42)

# Train the model
mlp_regressor.fit(X_train, y_train)

# Predict on validation set
predictions = mlp_regressor.predict(X_val)

# Calculate mean squared error
mse = mean_squared_error(y_val, predictions)
print(f"Mean Squared Error for MLPRegressor: {mse}")
