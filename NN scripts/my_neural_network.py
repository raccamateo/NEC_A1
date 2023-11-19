import numpy as np

# Activation Functions and Derivatives
def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative of ReLU activation function."""
    return (x > 0).astype(float)

def linear(x):
    """Linear activation function."""
    return x

def linear_derivative(x):
    """Derivative of linear activation function."""
    return np.ones_like(x)

class MyNeuralNetwork:
    def __init__(self, num_layers, units_per_layer, num_epochs, learning_rate, momentum, validation_split):
        """Initialize the neural network with given parameters."""
        self.L = num_layers  # Number of layers
        self.n = units_per_layer  # Units in each layer
        self.num_epochs = num_epochs  # Training epochs
        self.learning_rate = learning_rate  # Learning rate
        self.momentum = momentum  # Momentum coefficient
        self.validation_split = validation_split  # Validation data percentage

        # Initialize weights and biases
        self.w = [np.random.randn(self.n[l], self.n[l - 1]) * np.sqrt(2 / self.n[l - 1]) if l > 0 else None for l in range(self.L)]
        self.theta = [np.zeros((self.n[l], 1)) for l in range(1, self.L)]

        # Initialize previous weight and bias changes for momentum
        self.d_w_prev = [np.zeros_like(self.w[l]) if l > 0 else None for l in range(self.L)]
        self.d_theta_prev = [np.zeros_like(self.theta[l]) if l > 0 else None for l in range(self.L)]

        # Initialize lists to store errors for plotting
        self.training_errors = []
        self.validation_errors = []

    def _forward_pass(self, x):
        """Perform a forward pass through the network."""
        activations = [x.reshape(-1, 1)]  # Input layer activation
        # Forward propagate through the layers
        for l in range(1, self.L):
            z = np.dot(self.w[l], activations[l - 1]) + self.theta[l]
            # Apply ReLU for hidden layers, linear for the output layer
            activation = relu(z) if l < self.L - 1 else linear(z)
            activations.append(activation)
        return activations

    def _backward_pass(self, activations, y):
        """Perform backward propagation to compute gradients and update weights."""
        y = y.reshape(-1, 1)
        # Calculate output layer error
        delta = (activations[-1] - y) * linear_derivative(activations[-1])
        # Backpropagate the error and update weights and biases
        for l in range(self.L - 1, 0, -1):
            d_w = np.dot(delta, activations[l - 1].T)
            d_theta = delta
            # Apply momentum and learning rate to weight and bias updates
            self.w[l] -= self.learning_rate * d_w + self.momentum * self.d_w_prev[l]
            self.theta[l] -= self.learning_rate * d_theta + self.momentum * self.d_theta_prev[l]
            # Save updates for momentum
            self.d_w_prev[l] = d_w
            self.d_theta_prev[l] = d_theta
            if l > 1:
                # Calculate delta for the next layer (backpropagation)
                delta = np.dot(self.w[l].T, delta) * relu_derivative(activations[l - 1])

    def fit(self, X, y):
        """Train the network on the dataset for the specified number of epochs."""
        # Split data into training and validation sets
        split_at = int(len(X) * (1 - self.validation_split))
        X_train, X_val = X[:split_at], X[split_at:]
        y_train, y_val = y[:split_at], y[split_at:]

        for epoch in range(self.num_epochs):
            # Perform training for each sample
            for x, y_sample in zip(X_train, y_train):
                activations = self._forward_pass(x)
                self._backward_pass(activations, y_sample)
            
            # Calculate training error after each epoch
            train_error = self._compute_loss(X_train, y_train)
            self.training_errors.append(train_error)

            # Calculate validation error after each epoch if validation set exists
            if X_val.size > 0:
                val_error = self._compute_loss(X_val, y_val)
                self.validation_errors.append(val_error)

    def predict(self, X):
        """Predict outputs for given input data."""
        predictions = [self._forward_pass(x)[-1] for x in X]
        return np.squeeze(predictions)

    def loss_epochs(self):
        """Return the training and validation error for each epoch."""
        return np.array(self.training_errors), np.array(self.validation_errors)

    def _compute_loss(self, X, y):
        """Compute mean squared error loss."""
        predictions = self.predict(X)
        return np.mean((y - predictions) ** 2)

# Example usage:
# layers = [11, 9, 5, 1]  # Example architecture for the wine dataset
# nn = MyNeuralNetwork(num_layers=len(layers), units_per_layer=layers, num_epochs=100, learning_rate=0.01, momentum=0.9, validation_split=0.2)
# X and Y should be your data and labels arrays
# nn.fit(X, Y)
# training_errors, validation_errors = nn.loss_epochs()
# predictions = nn.predict(X_test)  # X_test is your test set
