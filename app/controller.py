from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model_filename = "model/water_potability_model.pkl"
with open(model_filename, "rb") as file:
    model = pickle.load(file)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    # The order of features should match the order used during model training
    feature_names = [
        "ph",
        "Hardness",
        "Solids",
        "Chloramines",
        "Sulfate",
        "Conductivity",
        "Organic_carbon",
        "Trihalomethanes",
        "Turbidity",
    ]
    try:
        # Extract features from the incoming JSON
        features = [data[feature] for feature in feature_names]
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)
        return jsonify(
            {
                "prediction": (
                    "Water is potable to drink"
                    if int(prediction[0]) == 1
                    else "Water is not potable to drink"
                )
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
