import pandas as pd
import joblib


# Load dataset
data = pd.read_csv("dataset/2022.csv")

# Clean column names
data.columns = data.columns.str.strip()

# Load trained ML model
model = joblib.load("model/rank2seat_model.pkl")


# -----------------------------
# Student Input
# -----------------------------

student_rank = int(input("\nEnter your JEE Rank: "))

print("\nAvailable Quotas:")
quotas = data["Quota"].dropna().unique()

for i, value in enumerate(quotas, start=1):
    print(f"{i}. {value}")

quota_choice = int(input("\nSelect Quota number: "))
quota = quotas[quota_choice - 1]


print("\nAvailable Seat Types:")
seat_types = data["Seat Type"].dropna().unique()

for i, value in enumerate(seat_types, start=1):
    print(f"{i}. {value}")

seat_choice = int(input("\nSelect Seat Type number: "))
seat_type = seat_types[seat_choice - 1]


print("\nAvailable Genders:")
genders = data["Gender"].dropna().unique()

for i, value in enumerate(genders, start=1):
    print(f"{i}. {value}")

gender_choice = int(input("\nSelect Gender number: "))
gender = genders[gender_choice - 1]


print("\nAvailable Rounds:")
rounds = sorted(data["Round"].dropna().unique())

for i, value in enumerate(rounds, start=1):
    print(f"{i}. Round {value}")

round_choice = int(input("\nSelect Round number: "))
round_number = rounds[round_choice - 1]


# -----------------------------
# Filter dataset
# -----------------------------

filtered_data = data[
    (data["Quota"] == quota) &
    (data["Seat Type"] == seat_type) &
    (data["Gender"] == gender) &
    (data["Round"] == round_number)
].copy()


# Check whether records exist
if filtered_data.empty:
    print("\nNo matching records found.")
    exit()


# -----------------------------
# Prepare ML input
# -----------------------------

X = filtered_data[
    [
        "Institute",
        "Academic Program Name",
        "Quota",
        "Seat Type",
        "Gender",
        "Round"
    ]
]


# Predict closing ranks
filtered_data["Predicted Closing Rank"] = model.predict(X)

filtered_data["Predicted Closing Rank"] = (
    filtered_data["Predicted Closing Rank"].round().astype(int)
)


# -----------------------------
# Calculate chance
# -----------------------------

def calculate_chance(rank, closing_rank):

    if rank <= closing_rank:
        return "GOOD"

    elif rank <= closing_rank * 1.10:
        return "MODERATE"

    else:
        return "LOW"


filtered_data["Chance"] = filtered_data.apply(
    lambda row: calculate_chance(
        student_rank,
        row["Predicted Closing Rank"]
    ),
    axis=1
)


# -----------------------------
# Keep useful recommendations
# -----------------------------

recommendations = filtered_data[
    filtered_data["Chance"].isin(["GOOD", "MODERATE"])
].copy()


# Sort by predicted closing rank
recommendations = recommendations.sort_values(
    by="Predicted Closing Rank",
    ascending=True
)


# -----------------------------
# Display results
# -----------------------------

print("\n")
print("=" * 70)
print("                 Rank2Seat Recommendations")
print("=" * 70)

print("Student JEE Rank :", student_rank)
print("Quota            :", quota)
print("Seat Type        :", seat_type)
print("Gender           :", gender)
print("Round            :", round_number)

print("\nRecommended Colleges and Programs:")
print("-" * 70)


if recommendations.empty:

    print("No suitable recommendations found.")

else:

    for index, row in recommendations.iterrows():

        print("\nInstitute       :", row["Institute"])

        print(
            "Academic Program:",
            row["Academic Program Name"]
        )

        print(
            "Opening Rank   :",
            row["Opening Rank"]
        )

        print(
            "Historical Closing Rank:",
            row["Closing Rank"]
        )

        print(
            "Predicted Closing Rank:",
            row["Predicted Closing Rank"]
        )

        print(
            "Chance          :",
            row["Chance"]
        )

        print("-" * 70)


print("\nShowing all suitable recommendations.")

print("=" * 70)