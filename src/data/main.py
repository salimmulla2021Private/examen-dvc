from data_loading import load_data
from data_processing import prep_data
from data_normalization import normalize_data
from model_best_parameters import save_best_parameters
from data_training import train_data
import pandas as pd
from model_evaluation import model_evaluation


if __name__ == "__main__":
    # Load your raw data
    raw_data = load_data("./data/raw_data/raw.csv")
    prep_data(raw_data)
    normalize_data("./data/processed_data")
    save_best_parameters("./data/processed_data")
    train_data("./data/processed_data")
    model_evaluation("./data/processed_data")
