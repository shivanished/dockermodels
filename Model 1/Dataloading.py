import pandas as pd
import os

# Get the directory of the current script
base_dir = os.path.dirname(__file__)

# Construct the absolute path to the CSV file
file_path = os.path.join(base_dir, 'diets/All_Diets.csv')

# Load the dataset
df = pd.read_csv(file_path)

# Inspect the dataset
print(df.head())
print(df.columns)

# Use the correct column name
print(df['Diet_type'].unique())  # Check the unique diet labels
