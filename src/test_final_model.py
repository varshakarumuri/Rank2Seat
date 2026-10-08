import pandas as pd
import joblib


# Load the saved final model
model = joblib.load("model/best_rank2seat_model.pkl")


# Load dataset to get valid values
data = pd.read_csv("dataset/2022.csv")

# Remove extra spaces from column names
data.columns = data.columns.str.strip()


# Take one existing row from the dataset
sample = data.iloc[0]


# Create input data
student_data = pd.DataFrame([{
    "Institute": sample["Institute"],
    "Academic Program Name": sample["Academic Program Name"],
    "Quota": sample["Quota"],
    "Seat Type": sample["Seat Type"],
    "Gender": sample["Gender"],
    "Round": sample["Round"]
}])


# Predict closing rank
prediction = model.predict(student_data)[0]


print("\n======================================")
print("       FINAL MODEL TEST")
print("======================================")

print("Institute:", sample["Institute"])
print("Academic Program:", sample["Academic Program Name"])
print("Quota:", sample["Quota"])
print("Seat Type:", sample["Seat Type"])
print("Gender:", sample["Gender"])
print("Round:", sample["Round"])

print("\nActual Closing Rank:", sample["Closing Rank"])
print("Predicted Closing Rank:", round(prediction))

print("======================================")