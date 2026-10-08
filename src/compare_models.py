import pandas as pd

# Load the JEE dataset
data = pd.read_csv("dataset/2022.csv")

# Remove extra spaces from column names
data.columns = data.columns.str.strip()

# Convert rank columns to numeric values
data["Opening Rank"] = pd.to_numeric(
    data["Opening Rank"],
    errors="coerce"
)

data["Closing Rank"] = pd.to_numeric(
    data["Closing Rank"],
    errors="coerce"
)

# Remove rows where rank values are missing
data = data.dropna(
    subset=["Opening Rank", "Closing Rank"]
)

print("Dataset loaded successfully.")
print("Rows:", len(data))
print("Columns:", len(data.columns))

print("\nColumns:")
print(data.columns.tolist())
# Features used to predict Closing Rank
features = [
    "Institute",
    "Academic Program Name",
    "Quota",
    "Seat Type",
    "Gender",
    "Round"
]

# Input features
X = data[features]

# Target value
y = data["Closing Rank"]

print("\nFeatures:")
print(features)

print("\nTarget:")
print("Closing Rank")

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)
from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nData split completed.")

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Categorical columns
categorical_columns = [
    "Institute",
    "Academic Program Name",
    "Quota",
    "Seat Type",
    "Gender"
]

# Create the preprocessor
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

print("\nPreprocessing setup completed.")
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Define regression models
models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        random_state=42
    )
}

print("\nRegression models created:")
for model_name in models:
    print("-", model_name)
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Store results of all models
results = []

# Train and evaluate each model
for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    # Create pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # Train model
    pipeline.fit(X_train, y_train)

    # Predict on test data
    predictions = pipeline.predict(X_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, predictions)

    mse = mean_squared_error(y_test, predictions)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, predictions)

    # Store results
    results.append({
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    print(f"{model_name} completed.")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.4f}")
# Convert results into a DataFrame
results_df = pd.DataFrame(results)

# Sort models by R2 Score
results_df = results_df.sort_values(
    by="R2 Score",
    ascending=False
)

print("\n======================================")
print("       REGRESSION MODEL RESULTS")
print("======================================")

print(
    results_df.to_string(
        index=False
    )
)

print("======================================")
# ==========================================
# CLASSIFICATION MODEL PREPARATION
# ==========================================

# Create closing-rank categories
# 0 = Low closing rank
# 1 = Medium closing rank
# 2 = High closing rank

data["Closing Rank Category"] = pd.qcut(
    data["Closing Rank"],
    q=3,
    labels=[0, 1, 2]
)

print("\nClosing Rank Category distribution:")
print(data["Closing Rank Category"].value_counts().sort_index())
# Classification features
X_class = data[features]

# Classification target
y_class = data["Closing Rank Category"].astype(int)

# Split classification data into training and testing sets
X_class_train, X_class_test, y_class_train, y_class_test = train_test_split(
    X_class,
    y_class,
    test_size=0.20,
    random_state=42,
    stratify=y_class
)

print("\nClassification data split completed.")

print("Classification training samples:", len(X_class_train))
print("Classification testing samples:", len(X_class_test))

print("\nClassification target values:")
print(sorted(y_class.unique()))
# ==========================================
# CLASSIFICATION MODELS
# ==========================================

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Define classification models
classification_models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Decision Tree Classifier": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest Classifier": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting Classifier": GradientBoostingClassifier(
        random_state=42
    )
}

print("\nClassification models created:")

for model_name in classification_models:
    print("-", model_name)
# ==========================================
# TRAIN AND EVALUATE CLASSIFICATION MODELS
# ==========================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

classification_results = []

for model_name, model in classification_models.items():

    print(f"\nTraining {model_name}...")

    # Create pipeline
    classification_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # Train the model
    classification_pipeline.fit(
        X_class_train,
        y_class_train
    )

    # Predict on test data
    predictions = classification_pipeline.predict(
        X_class_test
    )

    # Calculate metrics
    accuracy = accuracy_score(
        y_class_test,
        predictions
    )

    precision = precision_score(
        y_class_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_class_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_class_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    # Store results
    classification_results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print(f"{model_name} completed.")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
# ==========================================
# CLASSIFICATION MODEL RESULTS
# ==========================================

classification_results_df = pd.DataFrame(
    classification_results
)

# Sort by F1 Score
classification_results_df = (
    classification_results_df
    .sort_values(
        by="F1 Score",
        ascending=False
    )
)

print("\n==============================================")
print("        CLASSIFICATION MODEL RESULTS")
print("==============================================")

print(
    classification_results_df.to_string(
        index=False
    )
)

print("==============================================")