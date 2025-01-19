import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('BostonHousing.csv')

columns = [
    "crim", "zn", "indus", "chas", "nox", "rm", "age",
    "dis", "rad", "tax", "ptratio", "b", "lstat", "medv"
]

# Display the first few rows of the dataset
print("Dataset Preview:")
print(df.head())

# Separate features (X) and target (y)
X = df.drop(columns=['medv'])  # Features
y = df['medv']  # Target variable

# Normalize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Print dataset statistics
print("\nDataset Statistics:")
print(f"Total samples: {len(df)}")
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Model Training
# Linear Regression From Scratch

import numpy as np

class LinearRegressionFromScratch:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            # Calculate predictions
            y_pred = np.dot(X, self.weights) + self.bias

            # Compute gradients
            dw = -(2 / n_samples) * np.dot(X.T, (y - y_pred))
            db = -(2 / n_samples) * np.sum(y - y_pred)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Train the Linear Regression model
lr_model = LinearRegressionFromScratch(learning_rate=0.01, epochs=1000)
lr_model.fit(X_train, y_train)

# Predict on test data
y_pred_lr = lr_model.predict(X_test)


# Random Forest from Scratch
class RandomForestFromScratch:
    def __init__(self, n_trees=10, max_depth=5, sample_size=0.8, random_state=42):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.sample_size = sample_size
        self.random_state = random_state
        self.trees = []

    def _bootstrap_sample(self, X, y):
        n_samples = int(self.sample_size * X.shape[0])
        np.random.seed(self.random_state)
        indices = np.random.choice(X.shape[0], n_samples, replace=True)
        return X[indices], y.to_numpy()[indices]

    def _build_tree(self, X, y, depth=0):
        # A simple decision tree implementation (split data and return average for regression)
        if depth == self.max_depth or len(np.unique(y)) == 1:
            return np.mean(y)
        
        # Find the best split
        n_samples, n_features = X.shape
        best_mse = float("inf")
        best_split = None
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] <= threshold
                right_indices = X[:, feature] > threshold
                if len(np.unique(left_indices)) == 1 or len(np.unique(right_indices)) == 1:
                    continue

                left_y, right_y = y[left_indices], y[right_indices]
                mse = np.mean((left_y - np.mean(left_y))**2) + np.mean((right_y - np.mean(right_y))**2)

                if mse < best_mse:
                    best_mse = mse
                    best_split = (feature, threshold)
        
        if not best_split:
            return np.mean(y)

        # Recursive split
        feature, threshold = best_split
        left_indices = X[:, feature] <= threshold
        right_indices = X[:, feature] > threshold
        left_tree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        return (feature, threshold, left_tree, right_tree)

    def fit(self, X, y):
        self.trees = []
        self.feature_importances_ = np.zeros(X.shape[1])  # Initialize feature importance array
        for _ in range(self.n_trees):
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree = self._build_tree(X_sample, y_sample)
            self.trees.append(tree)

            # Update feature importance for this tree (example logic)
            for feature_idx in range(X.shape[1]):
                # Calculate feature importance (placeholder: update with actual logic)
                self.feature_importances_[feature_idx] += np.random.random()  # Replace this with proper calculation

        # Normalize feature importance
        self.feature_importances_ /= self.n_trees

    def _predict_tree(self, tree, x):
        if not isinstance(tree, tuple):
            return tree

        feature, threshold, left_tree, right_tree = tree
        if x[feature] <= threshold:
            return self._predict_tree(left_tree, x)
        else:
            return self._predict_tree(right_tree, x)

    def predict(self, X):
        predictions = np.array([self._predict_tree(tree, x) for tree in self.trees for x in X])
        return np.mean(predictions.reshape(self.n_trees, -1), axis=0)

# Train the Random Forest model
rf_model = RandomForestFromScratch(n_trees=10, max_depth=5)
rf_model.fit(X_train, y_train)

# Predict on test data
y_pred_rf = rf_model.predict(X_test)

# Performance Evaluation

from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Function to evaluate model performance
def evaluate_model(y_true, y_pred, model_name="Model"):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"Performance of {model_name}:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")
    return rmse, r2

# Example: Evaluate models (replace `y_pred_lr` and `y_pred_rf` with your predictions)
# Linear Regression
rmse_lr, r2_lr = evaluate_model(y_test, y_pred_lr, model_name="Linear Regression")

# Random Forest
rmse_rf, r2_rf = evaluate_model(y_test, y_pred_rf, model_name="Random Forest")

# XGBoost (if available)
# rmse_xgb, r2_xgb = evaluate_model(y_test, y_pred_xgb, model_name="XGBoost")

# Visualizing Feature Importance

import matplotlib.pyplot as plt

# Example: Feature importance for Random Forest (replace with your model's logic)
feature_importances = rf_model.feature_importances_  # Replace with actual implementation
features = df.columns[:-1]  # Exclude the target column

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(features, feature_importances, color='skyblue')
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Feature Importance - Random Forest")
plt.show()
