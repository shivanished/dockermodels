import pandas as pd
import os

# Get the directory of the current script
base_dir = os.path.dirname(__file__)

# Construct the absolute path to the CSV file
file_path = os.path.join(base_dir, 'TasteTrios - Sheet1.csv')

# Load the dataset
df = pd.read_csv(file_path)

# Inspect the dataset
print(df.head())
print(df.columns)
print(df.info())
