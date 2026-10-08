import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline


# ==========================================
# 1. LOAD DATASET
# ==========================================

data = pd.read_csv("dataset/2022.csv")

# Clean column names
data.columns = data.columns.str.strip()


# ==========================================
# 2. CLEAN RANK COLUMNS
# ==========================================

data["Opening Rank"] = pd.to_numeric(
    data["Opening Rank"],
    errors="coerce"
)

data["Closing Rank"] = pd.to_numeric(
    data["Closing Rank"],
    errors="coerce"
)

# Remove invalid rows
data = data.dropna(
    subset=["Opening Rank", "Closing Rank"]
)


# ==========================================
# 3. SELECT FEATURES
# ==========================================

features = [
    "Institute",
    "Academic Program Name",
    "Quota",
    "Seat Type",
    "Gender",
    "Round"
]

X = data[features]

# Target
y = data["Closing Rank"]


# ==========================================
# 4. CATEGORICAL PREPROCESSING
# ==========================================

categorical_columns = [
    "Institute",
    "Academic Program Name",
    "Quota",
    "Seat Type",
    "Gender"
]

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


# ==========================================
# 5. CREATE FINAL RANDOM FOREST MODEL
# ==========================================

random_forest = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


# ==========================================
# 6. CREATE PIPELINE
# ==========================================

final_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", random_forest)
    ]
)


# ==========================================
# 7. TRAIN ON COMPLETE DATASET
# ==========================================

print("Training final Random Forest model...")

final_pipeline.fit(X, y)

print("Final model training completed!")


# ==========================================
# 8. SAVE FINAL MODEL
# ==========================================

joblib.dump(
    final_pipeline,
    "model/best_rank2seat_model.pkl"
)

print("Best model saved successfully!")
print("Location: model/best_rank2seat_model.pkl")