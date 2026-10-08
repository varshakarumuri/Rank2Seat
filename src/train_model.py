import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load the dataset
data = pd.read_csv("dataset/2022.csv")

# Clean column names
data.columns = data.columns.str.strip()

# Convert rank columns to numbers
data["Opening Rank"] = pd.to_numeric(
    data["Opening Rank"], errors="coerce"
)

data["Closing Rank"] = pd.to_numeric(
    data["Closing Rank"], errors="coerce"
)

# Remove rows with missing rank values
data = data.dropna(
    subset=["Opening Rank", "Closing Rank"]
)


# Features
X = data[
    [
        "Institute",
        "Academic Program Name",
        "Quota",
        "Seat Type",
        "Gender",
        "Round"
    ]
]

# Target
y = data["Closing Rank"]


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Categorical columns
categorical_columns = [
    "Institute",
    "Academic Program Name",
    "Quota",
    "Seat Type",
    "Gender"
]


# Convert categorical data into numbers
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder="passthrough"
)


# Create ML model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


# Create complete ML pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# Train the model
pipeline.fit(X_train, y_train)


# Make predictions
predictions = pipeline.predict(X_test)


# Evaluate the model
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)


print("\nModel Training Completed!")

print("\nModel Evaluation:")
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)
# Save the trained model
joblib.dump(pipeline, "model/rank2seat_model.pkl")

print("\nModel saved successfully!")