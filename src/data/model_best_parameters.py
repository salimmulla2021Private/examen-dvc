
from pathlib import Path
from typing import IO

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


def save_best_parameters(data_path: str | Path | IO[bytes]) -> pd.DataFrame:
    """
    Decide on the regression model to implement GridSearch for Best Parameters and the parameters to test. 
    At the end of this script, we will have the best parameters saved as a .pkl file in the models directory
    """
    params = {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
    }

    X_train_scaled = pd.read_csv(f"{data_path}/X_train_scaled.csv")
    y_train = pd.read_csv(f"{data_path}/y_train.csv")

    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(),
        param_grid=params,
        cv=5,
        n_jobs=-1,
        verbose=2
    )
    grid_search.fit(X_train_scaled, y_train)

    save_path = Path("./models/best_parameters.pkl")
    pd.to_pickle(pd.DataFrame([grid_search.best_params_]), save_path)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best parameters saved to {save_path}")



    
