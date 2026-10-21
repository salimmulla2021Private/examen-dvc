
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


def model_evaluation(data_path: str | Path | IO[bytes]):
    """
    Finally, using the trained model, we will evaluate its performance and make predictions. 
    At the end of this script, we will have a new dataset in data containing the predictions, 
    along with a scores.json file in the metrics directory that will capture evaluation metrics 
    of our model (e.g., MSE, R2).
    """
    
    X_test_scaled = pd.read_csv(f"{data_path}/X_test_scaled.csv")

    y_test = pd.read_csv(f"{data_path}/y_test.csv")

    # Load the trained model
    model = pd.read_pickle("./models/trained_model.pkl")

    # Make predictions
    predictions = model.predict(X_test_scaled)

    # Evaluate the model
    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Save the evaluation metrics
    scores = {"MSE": mse, "R2": r2}
    scores_path = Path("./metrics/scores.json")
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    pd.Series(scores).to_json(scores_path)
    logging.info(f"Evaluation metrics saved to {scores_path}")
    print(scores)
