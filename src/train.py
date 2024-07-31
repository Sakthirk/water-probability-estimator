import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from collections import Counter
import pickle
import logging
import mlflow
import mlflow.sklearn

# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def import_data(filepath):
    """
    Imports data from a CSV file.
    Args:
        filepath (str): Path to the CSV file.
    Returns:
        DataFrame: Pandas DataFrame containing the imported data.
    """
    try:
        data = pd.read_csv(filepath)
        logging.info("Data imported successfully from %s", filepath)
        return data
    except Exception as e:
        logging.error("Error importing data: %s", e)
        raise

def preprocess_data(data):
    """
    Fills missing values in specified columns with the mean of the group based on 'Potability'.
    Args:
        data (DataFrame): The pandas DataFrame to preprocess.
    Returns:
        DataFrame: The preprocessed pandas DataFrame.
    """
    try:
        for column in ["ph", "Sulfate", "Trihalomethanes"]:
            data[column] = data[column].fillna(
                data.groupby(["Potability"])[column].transform("mean")
            )
        logging.info("Data preprocessing completed")
        return data
    except Exception as e:
        logging.error("Error in data preprocessing: %s", e)
        raise

def balance_data(X, y):
    """
    Balances the data using SMOTE to handle class imbalance.
    Args:
        X (DataFrame): Features dataset.
        y (Series): Target dataset.
    Returns:
        tuple: The resampled features and target datasets.
    """
    try:
        smt = SMOTE()
        logging.info("Before SMOTE: %s", Counter(y))
        X_res, y_res = smt.fit_resample(X, y)
        logging.info("After SMOTE: %s", Counter(y_res))
        return X_res, y_res
    except Exception as e:
        logging.error("Error in balancing data: %s", e)
        raise

def scale_features(X):
    """
    Scales features using StandardScaler.
    Args:
        X (DataFrame): Dataset to scale.
    Returns:
        tuple: Scaled dataset and the scaler object used.
    """
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        logging.info("Feature scaling completed")
        return X_scaled, scaler
    except Exception as e:
        logging.error("Error in scaling features: %s", e)
        raise

def train_logistic_regression(X, y, penalty, C):
    """
    Trains a logistic regression model.
    Args:
        X (DataFrame): Input features for training.
        y (Series): Target labels for training.
        penalty (str): Penalty type ('l1' or 'l2').
        C (float): Inverse of regularization strength.
    Returns:
        LogisticRegression: The trained logistic regression model.
    """
    try:
        model = LogisticRegression(
            penalty=penalty,
            C=C,
            random_state=42,
            max_iter=10000,
            solver="liblinear",
        )
        model.fit(X, y)
        logging.info("Model trained with penalty=%s, C=%s", penalty, C)
        return model
    except Exception as e:
        logging.error("Error in training model: %s", e)
        raise

def save_model(model, filename):
    """
    Saves the logistic regression model to a file using pickle.
    Args:
        model (LogisticRegression): The logistic regression model to save.
        filename (str): The filename to save the model to.
    """
    try:
        dir_name = filename.split("/")[0]
        os.makedirs(dir_name, exist_ok=True)
        with open(filename, "wb") as file:
            pickle.dump(model, file)
        logging.info("Model saved to %s", filename)
    except Exception as e:
        logging.error("Error saving model: %s", e)
        raise

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the logistic regression model using the test dataset and prints accuracy and classification report.
    Args:
        model (LogisticRegression): The model to evaluate.
        X_test (DataFrame): Test features dataset.
        y_test (Series): Test target dataset.
    """
    try:
        y_pred = model.predict(X_test)
        acc_score = accuracy_score(y_test, y_pred)
        logging.info("Test Accuracy: %s", acc_score)
        logging.info("\n%s", classification_report(y_test, y_pred))
        return acc_score
    except Exception as e:
        logging.error("Error evaluating model: %s", e)
        raise

def main():
    """
    Main function to run the model training and evaluation pipeline.
    """
    mlflow.sklearn.autolog()  # Enable autologging for sklearn

    data = import_data("data/water_potability.csv")
    data = preprocess_data(data)
    X = data.drop("Potability", axis=1)
    y = data["Potability"]
    X_resampled, y_resampled = balance_data(X, y)
    X_scaled, scaler = scale_features(X_resampled)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_resampled, test_size=0.20, random_state=42
    )

    penalties = ["l1", "l2"]
    C_values = [1, 0.5, 0.1]

    for penalty in penalties:
        for C in C_values:
            with mlflow.start_run():
                model = train_logistic_regression(X_train, y_train, penalty, C)
                acc_score = evaluate_model(model, X_test, y_test)
                mlflow.log_param("penalty", penalty)
                mlflow.log_param("C", C)
                mlflow.log_metric("accuracy", acc_score)
                mlflow.log_artifact("data/water_potability.csv")

    save_model(model, "model/water_potability_model.pkl")

if __name__ == "__main__":
    main()
