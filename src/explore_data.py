import pandas as pd

# Load the JEE dataset
data = pd.read_csv("dataset/2022.csv")

print("Number of rows:", len(data))
print("Number of columns:", len(data.columns))

print("\nColumn names:")
print(data.columns.tolist())

print("\nFirst 5 rows:")
print(data.head())

print("\nMissing values:")
print(data.isnull().sum())