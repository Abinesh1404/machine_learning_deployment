import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import joblib

# Assuming 'app', 'model', and 'scaler' are already defined and loaded from previous steps
# If running this cell independently, ensure the following are loaded:
# app = Flask(__name__)
# model = joblib.load('logistic_regression_model.pkl')
# scaler = joblib.load('scaler.pkl')

# Define the column names based on the training data (assuming X from previous steps)
# In a real API, these column names would be fixed or part of the model metadata
feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

@app.route('/predict', methods=['POST'])
def predict():
    if not request.json:
        return jsonify({"error": "Invalid input, request must be JSON"}), 400

    try:
        # Get JSON data from the request
        data = request.json

        # Convert the received JSON data into a pandas DataFrame
        # Ensure the DataFrame has the same column order as the training data
        input_df = pd.DataFrame([data])
        input_df = input_df[feature_columns] # Ensure correct column order

        # Use the loaded scaler to transform the input data
        scaled_data = scaler.transform(input_df)

        # Use the loaded model to make a prediction
        prediction = model.predict(scaled_data)[0]
        prediction_proba = model.predict_proba(scaled_data)[0].tolist()

        # Convert prediction to human-readable format
        result = {
            "prediction": int(prediction), # 0 or 1
            "probability_no_diabetes": prediction_proba[0],
            "probability_diabetes": prediction_proba[1]
        }

        return jsonify(result)

    except KeyError as e:
        return jsonify({"error": f"Missing required feature in input: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

print("'/predict' endpoint defined.")

