import pandas as pd

# Load dataset
data = pd.read_csv("dataset/2022.csv")

# Display basic statistics
print("\nDataset Description:")
print(data.describe())

# Display unique values in categorical columns
print("\nUnique Quotas:")
print(data["Quota"].unique())

print("\nUnique Seat Types:")
print(data["Seat Type"].unique())

print("\nUnique Genders:")
print(data["Gender"].unique())

print("\nNumber of Institutes:")
print(data["Institute"].nunique())

print("\nNumber of Academic Programs:")
print(data["Academic Program Name"].nunique())

print("\nNumber of Rounds:")
print(data["Round"].unique())

print("\nYears Available:")
print(data["Year"].unique())