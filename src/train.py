import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from collections import Counter
import pickle


def import_data(filepath):
    """
    Imports data from a CSV file.
    Args:
        filepath (str): Path to the CSV file.
    Returns:
        DataFrame: Pandas DataFrame containing the imported data.
    """
    return pd.read_csv(filepath)


def preprocess_data(data):
    """
    Fills missing values in specified columns with the mean of the group based on 'Potability'.
    Args:
        data (DataFrame): The pandas DataFrame to preprocess.
    Returns:
        DataFrame: The preprocessed pandas DataFrame.
    """
    for column in ["ph", "Sulfate", "Trihalomethanes"]:
        data[column] = data[column].fillna(
            data.groupby(["Potability"])[column].transform("mean")
        )
    return data


def balance_data(X, y):
    """
    Balances the data using SMOTE to handle class imbalance.
    Args:
        X (DataFrame): Features dataset.
        y (Series): Target dataset.
    Returns:
        tuple: The resampled features and target datasets.
    """
    print("Balancing the data by SMOTE - Oversampling of Minority level")
    smt = SMOTE()
    print("Before SMOTE", Counter(y))
    X_res, y_res = smt.fit_resample(X, y)
    print("After SMOTE", Counter(y_res))
    return X_res, y_res


def scale_features(X):
    """
    Scales features using StandardScaler.
    Args:
        X (DataFrame): Dataset to scale.
    Returns:
        tuple: Scaled dataset and the scaler object used.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def train_logistic_regression(X, y):
    """
    Trains a logistic regression model by iterating over combinations of penalties and C values to find the best model.
    Args:
        X (DataFrame): Input features for training.
        y (Series): Target labels for training.
    Returns:
        tuple: Best logistic regression model, penalty, C value, and accuracy.
    """
    C_values = [1, 0.5, 0.1, 0.003, 0.01]
    best_model = None
    best_accuracy = 0
    best_penalty = ""
    best_C = 0
    for penalty in ["l1", "l2"]:
        for C in C_values:
            model = LogisticRegression(
                penalty=penalty,
                C=C,
                random_state=42,
                max_iter=10000,
                solver="liblinear",
            )
            model.fit(X, y)
            accuracy = accuracy_score(y, model.predict(X))
            print(f"Penalty: {penalty}, C: {C}, Training Accuracy: {accuracy}")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_penalty = penalty
                best_C = C
                best_model = model
    print(
        f"Best Penalty: {best_penalty}, Best C: {best_C}, Best Training Accuracy: {best_accuracy}"
    )
    return best_model, best_penalty, best_C, best_accuracy


def save_model(model, filename):
    """
    Saves the logistic regression model to a file using pickle.
    Args:
        model (LogisticRegression): The logistic regression model to save.
        filename (str): The filename to save the model to.
    """
    dir_name = filename.split("/")[0]
    os.makedirs(dir_name, exist_ok=True)
    with open(filename, "wb") as file:
        pickle.dump(model, file)


def evaluate_model(model, X_test, y_test):
    """
    Evaluates the logistic regression model using the test dataset and prints accuracy and classification report.
    Args:
        model (LogisticRegression): The model to evaluate.
        X_test (DataFrame): Test features dataset.
        y_test (Series): Test target dataset.
    """
    y_pred = model.predict(X_test)
    acc_score = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc_score}")
    print(classification_report(y_test, y_pred))


def main():
    """
    Main function to run the model training and evaluation pipeline.
    """
    data = import_data("data/water_potability.csv")
    data = preprocess_data(data)
    X = data.drop("Potability", axis=1)
    y = data["Potability"]
    X_resampled, y_resampled = balance_data(X, y)
    X_scaled, scaler = scale_features(X_resampled)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_resampled, test_size=0.20
    )
    model, _, _, _ = train_logistic_regression(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model, "model/water_potability_model.pkl")


if __name__ == "__main__":
    main()
