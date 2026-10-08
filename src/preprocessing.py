import pandas as pd

# Load the JEE dataset
data = pd.read_csv("dataset/2022.csv")

# Remove extra spaces from column names
data.columns = data.columns.str.strip()

# Convert rank columns to numeric values
data["Opening Rank"] = pd.to_numeric(data["Opening Rank"], errors="coerce")
data["Closing Rank"] = pd.to_numeric(data["Closing Rank"], errors="coerce")

# Remove rows where rank information is missing
data = data.dropna(subset=["Opening Rank", "Closing Rank"])

# Convert ranks to integers
data["Opening Rank"] = data["Opening Rank"].astype(int)
data["Closing Rank"] = data["Closing Rank"].astype(int)

print("Preprocessing completed successfully.")

print("\nDataset shape:")
print(data.shape)

print("\nFirst 5 rows after preprocessing:")
print(data.head())

print("\nData types:")
print(data.dtypes)