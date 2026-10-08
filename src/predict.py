import pandas as pd
import joblib

# Load dataset
data = pd.read_csv("dataset/2022.csv")

# Load trained model
model = joblib.load("model/rank2seat_model.pkl")


# Function to display choices
def show_choices(column_name):
    print(f"\nAvailable {column_name}:")
    values = data[column_name].dropna().unique()

    for i, value in enumerate(values, start=1):
        print(f"{i}. {value}")

    return values


# Student's JEE rank
student_rank = int(input("\nEnter your JEE Rank: "))


# Institute
institutes = show_choices("Institute")
institute_choice = int(input("\nSelect Institute number: "))
institute = institutes[institute_choice - 1]


# Academic Program
programs = show_choices("Academic Program Name")
program_choice = int(input("\nSelect Academic Program number: "))
program = programs[program_choice - 1]


# Quota
quotas = show_choices("Quota")
quota_choice = int(input("\nSelect Quota number: "))
quota = quotas[quota_choice - 1]


# Seat Type
seat_types = show_choices("Seat Type")
seat_choice = int(input("\nSelect Seat Type number: "))
seat_type = seat_types[seat_choice - 1]


# Gender
genders = show_choices("Gender")
gender_choice = int(input("\nSelect Gender number: "))
gender = genders[gender_choice - 1]


# Round
round_number = int(input("\nEnter Round (1-6): "))


# Create input for ML model
student_data = pd.DataFrame([{
    "Institute": institute,
    "Academic Program Name": program,
    "Quota": quota,
    "Seat Type": seat_type,
    "Gender": gender,
    "Round": round_number
}])


# Predict closing rank
predicted_closing_rank = model.predict(student_data)[0]

predicted_closing_rank = round(predicted_closing_rank)


# Display result
print("\n================================")
print("       Rank2Seat Prediction")
print("================================")

print("Student JEE Rank:", student_rank)
print("Institute:", institute)
print("Academic Program:", program)
print("Quota:", quota)
print("Seat Type:", seat_type)
print("Gender:", gender)
print("Round:", round_number)

print("\nPredicted Closing Rank:", predicted_closing_rank)


# Chance calculation
if student_rank <= predicted_closing_rank:
    print("Chance: GOOD")
else:
    print("Chance: LOW")

print("================================")