
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prep_data(data: pd.DataFrame):
    """
    Validate and preprocess mineral processing data. We will work with a dataset containing information about mineral processing, 
    specifically focusing on the flotation process used to concentrate silica from ore.
    """

    df = data.dropna().copy()

    if df.empty:
        raise ValueError("CSV has no complete rows to process.")
    print(df.head())

    df = df.drop(columns='date')
    X = df.drop(columns="silica_concentrate")
    y = df["silica_concentrate"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

    # Save dataframes to their respective output file paths
    save_dataframes(X_train, X_test, y_train, y_test, "./data/processed_data")


def save_dataframes(X_train, X_test, y_train, y_test, output_folderpath):
    # Save dataframes to their respective output file paths
    for file, filename in zip([X_train, X_test, y_train, y_test], ['X_train', 'X_test', 'y_train', 'y_test']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        if check_existing_file(output_filepath):
            file.to_csv(output_filepath, index=False)
            print(f"Saved data to {output_filepath}")


def check_existing_file(file_path):
    '''Check if a file already exists. If it does, ask if we want to overwrite it.'''
    if os.path.isfile(file_path):
        while True:
            response = input(f"File {os.path.basename(file_path)} already exists. Do you want to overwrite it? (y/n): ")
            if response.lower() == 'y':
                return True
            elif response.lower() == 'n':
                return False
            else:
                print("Invalid response. Please enter 'y' or 'n'.")
    else:
        return True