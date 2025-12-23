import numpy as np
from flask import Flask, request, jsonify
import joblib

# ✅ CREATE FLASK APP FIRST
app = Flask(__name__)

# Load trained model and scaler
model = joblib.load('logistic_regression_model.pkl')
scaler = joblib.load('scaler.pkl')

# Feature order (must match training)
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
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    try:
        data = request.get_json()

        # Check missing features
        missing = [f for f in feature_columns if f not in data]
        if missing:
            return jsonify({"error": f"Missing features: {missing}"}), 400

        # Convert input to NumPy array
        input_array = np.array(
            [[float(data[f]) for f in feature_columns]]
        )

        # Scale and predict
        scaled_input = scaler.transform(input_array)
        prediction = int(model.predict(scaled_input)[0])
        probabilities = model.predict_proba(scaled_input)[0]

        return jsonify({
            "prediction": prediction,
            "probability_no_diabetes": float(probabilities[0]),
            "probability_diabetes": float(probabilities[1])
        })

    except ValueError:
        return jsonify({"error": "All input values must be numeric"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ✅ RUN FLASK APP
if __name__ == "__main__":
    app.run(debug=True)


