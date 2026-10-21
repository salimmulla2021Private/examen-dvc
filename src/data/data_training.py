
# Imports librairies

from pathlib import Path
from typing import IO

from matplotlib.pyplot import clf
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np
from scipy.stats import randint, test
from sklearn.linear_model  import LogisticRegression 
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import logging


logging.basicConfig(level=logging.INFO)


def train_data(data_path: str | Path | IO[bytes]):
    """
    Using the parameters found through GridSearch, we will train the model and 
    save the trained model in the models directory.
    """
    
    X_train_scaled = pd.read_csv(f"{data_path}/X_train_scaled.csv")

    y_train = pd.read_csv(f"{data_path}/y_train.csv")

    # Load the best parameters
    best_params = pd.read_pickle("./models/best_parameters.pkl").iloc[0].to_dict()

    # Initialize the model with the best parameters
    model = RandomForestRegressor(**best_params)

    # Train the model
    model.fit(X_train_scaled, y_train)

    # Save the trained model
    save_path = Path("./models/trained_model.pkl")
    pd.to_pickle(model, save_path)
    logging.info(f"Trained model saved to {save_path}")


def main():
    train_data("./data/processed")

if __name__ == "__main__":
    main()