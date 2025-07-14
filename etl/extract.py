# extract.py - Auto-generated
def read_specific_file(file_path):
    import pandas as pd
    import os
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist.")
    df = pd.read_csv(file_path)
    print(f" Loaded: {file_path}")
    return df