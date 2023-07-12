import numpy as np
import joblib
import warnings

# Load the model from the file
model = joblib.load('diabetes_model.joblib')

# Example input data for prediction (with 13 features)
new_data = np.array([3,126,88,41,235,39.3,0.704,27
])

# Reshape the input data to a 2D array with the same number of features
new_data_reshaped = new_data.reshape(1, -1)

# Make predictions using the loaded model
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    predictions = model.predict(new_data_reshaped)

print(predictions)
