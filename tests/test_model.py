import unittest
import pickle
import numpy as np


class WaterPotabilityModel:
    def __init__(self, model_file):
        # Load the trained logistic regression model
        with open(model_file, "rb") as file:
            self.model = pickle.load(file)

    def predict(self, features):
        # Predict potability using the model
        features_array = np.array(features).reshape(1, -1)
        return self.model.predict(features_array)[0]


class TestWaterPotabilityModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize the model once for all tests
        cls.model = WaterPotabilityModel("model/water_potability_model.pkl")

    def test_not_potable_water(self):
        # Features corresponding to not potable water
        input_data = {
            "ph": 3.71608,
            "Hardness": 129.422921,
            "Solids": 18630.057858,
            "Chloramines": 6.635246,
            "Sulfate": 1000000000,
            "Conductivity": 592.885359,
            "Organic_carbon": 15.180013,
            "Trihalomethanes": 56.329076,
            "Turbidity": 4.500656,
        }
        features = [input_data[key] for key in input_data]
        prediction = self.model.predict(features)
        self.assertEqual(prediction, 0, "The water should be predicted as not potable.")

    def test_potable_water(self):
        # Features corresponding to potable water
        input_data = {
            "ph": 7.0,
            "Hardness": 150,
            "Solids": 15000,
            "Chloramines": 7,
            "Sulfate": 250,
            "Conductivity": 400,
            "Organic_carbon": 15,
            "Trihalomethanes": 70,
            "Turbidity": 3.5,
        }
        features = [input_data[key] for key in input_data]
        prediction = self.model.predict(features)
        self.assertEqual(prediction, 1, "The water should be predicted as potable.")


if __name__ == "__main__":
    unittest.main()
