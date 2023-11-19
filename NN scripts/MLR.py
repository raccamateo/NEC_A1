from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the dataset from the provided URL
url = "https://raw.githubusercontent.com/raccamateo/NEC_BP_LR/main/normalized_wine_data.csv"
wine_data = pd.read_csv(url)
print(wine_data.head())  # Just print the first few rows to check


X = wine_data.drop('quality', axis=1)
y = wine_data['quality']

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the LinearRegression model
linear_regressor = LinearRegression()

# Train the model using the training data
linear_regressor.fit(X_train, y_train)

# Predict the target values for the validation set
y_pred_val = linear_regressor.predict(X_val)

# Calculate the Mean Squared Error (MSE) on the validation set
mse = mean_squared_error(y_val, y_pred_val)
print(f"Mean Squared Error for Multiple Linear Regression: {mse}")
