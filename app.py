import numpy as np
from flask import Flask, request, jsonify
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load trained model and scaler
model = joblib.load('logistic_regression_model.pkl')
scaler = joblib.load('scaler.pkl')

# Feature order must match training data
feature_columns = [
    'Pregnancies',
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age'
]

@app.route('/predict', methods=['POST'])
def predict():
    # Ensure JSON request
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    try:
        data = request.get_json()

        # Validate required features
        missing_features = [f for f in feature_columns if f not in data]
        if missing_features:
            return jsonify({
                "error": f"Missing required features: {missing_features}"
            }), 400

        # Convert input JSON → NumPy array (2D)
        input_array = np.array(
            [[float(data[feature]) for feature in feature_columns]]
        )

        # Scale input
        scaled_data = scaler.transform(input_array)

        # Predict
        prediction = int(model.predict(scaled_data)[0])
        prediction_proba = model.predict_proba(scaled_data)[0]

        # Response
        result = {
            "prediction": prediction,  # 0 = No Diabetes, 1 = Diabetes
            "probability_no_diabetes": float(prediction_proba[0]),
            "probability_diabetes": float(prediction_proba[1])
        }

        return jsonify(result), 200

    except ValueError:
        return jsonify({"error": "All input values must be numeric"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)

