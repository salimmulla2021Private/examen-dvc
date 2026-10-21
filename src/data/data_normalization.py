
import os
from typing import IO

from anyio import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def normalize_data(data_path: str | Path | IO[bytes]):
    """
    Normalize the training and testing feature data using StandardScaler.
    The normalized data will be saved to the specified output folder.
    """

    # Save dataframes to their respective output file paths
    X_train = pd.read_csv(f"{data_path}/X_train.csv")
    X_test  = pd.read_csv(f"{data_path}/X_test.csv")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    save_scaled_data(X_train_scaled, X_test_scaled, data_path)


def save_scaled_data(X_train_scaled, X_test_scaled, output_folderpath):
    for file, filename in zip([X_train_scaled, X_test_scaled], ['X_train_scaled', 'X_test_scaled']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        if check_existing_file(output_filepath):
            pd.DataFrame(file).to_csv(output_filepath, index=False)
        print(f"Saved scaled data to {output_filepath}")


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