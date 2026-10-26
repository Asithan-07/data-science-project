
# ============================================================
# PROJECT: PREDICT CUSTOMER CHURN FOR A BANKING SYSTEM
# Course: Data Science and Statistical Analysis
# Algorithm: Random Forest Classifier
# ============================================================

# 1. IMPORT REQUIRED LIBRARIES

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Set visualization style
sns.set_theme(style="whitegrid")

# Create output directory
os.makedirs("outputs", exist_ok=True)

print("=" * 60)
print("BANK CUSTOMER CHURN PREDICTION")
print("=" * 60)


# 2. LOAD DATASET

df = pd.read_csv("Churn_Modelling.csv")

print("\n1. DATASET INFORMATION")
print("-" * 40)

# Remove unnecessary identifier columns if present
df.drop(
    columns=["RowNumber", "CustomerId", "Surname"],
    errors="ignore",
    inplace=True
)

print("Dataset Shape:", df.shape)

print("\nFirst Five Records:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:", df.duplicated().sum())


# 3. DATA CLEANING

df.drop_duplicates(inplace=True)

target = "Exited"

if target not in df.columns:
    raise ValueError(
        "The target column 'Exited' was not found. "
        "Please verify your dataset."
    )

# Check target values
print("\nTarget Variable Distribution:")
print(df[target].value_counts(dropna=False))

# Remove records with a missing target
df = df.dropna(subset=[target])

# Ensure the target contains the expected binary classes
if not set(df[target].unique()).issubset({0, 1}):
    raise ValueError(
        "The Exited column must contain binary values 0 and 1."
    )

df[target] = df[target].astype(int)

X = df.drop(columns=[target])
y = df[target]


# 4. EXPLORATORY DATA ANALYSIS

print("\n2. EXPLORATORY DATA ANALYSIS")
print("-" * 40)

print("\nChurn Counts:")
print(y.value_counts())

print("\nChurn Percentage:")
print((y.value_counts(normalize=True) * 100).round(2))


# GRAPH 1: CUSTOMER CHURN DISTRIBUTION

plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="Exited")

plt.title("Customer Churn Distribution")
plt.xlabel("Exited (0 = Stayed, 1 = Churned)")
plt.ylabel("Number of Customers")
plt.tight_layout()

plt.savefig("outputs/1_churn_distribution.png", dpi=300)
plt.show()


# GRAPH 2: CHURN BY GEOGRAPHY

if "Geography" in df.columns:

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x="Geography",
        hue="Exited"
    )

    plt.title("Customer Churn by Geography")
    plt.xlabel("Geography")
    plt.ylabel("Number of Customers")
    plt.tight_layout()

    plt.savefig("outputs/2_churn_by_geography.png", dpi=300)
    plt.show()


# GRAPH 3: AGE DISTRIBUTION BY CHURN

if "Age" in df.columns:

    plt.figure(figsize=(8, 5))

    sns.histplot(
        data=df,
        x="Age",
        hue="Exited",
        bins=20,
        multiple="stack"
    )

    plt.title("Age Distribution by Customer Churn")
    plt.xlabel("Age")
    plt.ylabel("Number of Customers")
    plt.tight_layout()

    plt.savefig("outputs/3_age_distribution.png", dpi=300)
    plt.show()


# GRAPH 4: CHURN BY NUMBER OF PRODUCTS

if "NumOfProducts" in df.columns:

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x="NumOfProducts",
        hue="Exited"
    )

    plt.title("Customer Churn by Number of Products")
    plt.xlabel("Number of Products")
    plt.ylabel("Number of Customers")
    plt.tight_layout()

    plt.savefig("outputs/4_churn_by_products.png", dpi=300)
    plt.show()


# 5. TRAIN-TEST SPLIT

print("\n3. SPLITTING THE DATASET")
print("-" * 40)

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))


# 6. DATA PREPROCESSING

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# 7. BUILD RANDOM FOREST MODEL

print("\n4. MODEL TRAINING")
print("-" * 40)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1
            )
        )
    ]
)

model.fit(X_train, y_train)

print("Random Forest Model Trained Successfully!")


# 8. PREDICT CUSTOMER CHURN

y_pred = model.predict(X_test)


# 9. MODEL EVALUATION

print("\n5. MODEL EVALUATION")
print("-" * 40)

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, zero_division=0
)

recall = recall_score(
    y_test, y_pred, zero_division=0
)

f1 = f1_score(
    y_test, y_pred, zero_division=0
)

print(f"Accuracy  : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Stayed", "Churned"],
        zero_division=0
    )
)


# Save actual evaluation results

with open("outputs/model_results.txt", "w", encoding="utf-8") as file:

    file.write("BANK CUSTOMER CHURN PREDICTION\n")
    file.write("=" * 40 + "\n")

    file.write(f"Dataset records: {len(df)}\n")
    file.write(f"Training records: {len(X_train)}\n")
    file.write(f"Testing records: {len(X_test)}\n")

    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"Precision: {precision:.4f}\n")
    file.write(f"Recall: {recall:.4f}\n")
    file.write(f"F1-Score: {f1:.4f}\n\n")

    file.write(
        classification_report(
            y_test,
            y_pred,
            target_names=["Stayed", "Churned"],
            zero_division=0
        )
    )


# GRAPH 5: CONFUSION MATRIX

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Stayed", "Churned"],
    yticklabels=["Stayed", "Churned"]
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()

plt.savefig("outputs/5_confusion_matrix.png", dpi=300)
plt.show()


# 10. FEATURE IMPORTANCE

print("\n6. FEATURE IMPORTANCE")
print("-" * 40)

classifier = model.named_steps["classifier"]

feature_names = (
    model.named_steps["preprocessor"]
    .get_feature_names_out()
)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": classifier.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features:")
print(importance_df.head(10).to_string(index=False))


# GRAPH 6: FEATURE IMPORTANCE

top_features = importance_df.head(10).sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=top_features,
    x="Importance",
    y="Feature"
)

plt.title("Top 10 Features Used by the Model")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.tight_layout()

plt.savefig("outputs/6_feature_importance.png", dpi=300)
plt.show()


# 11. SAVE PREDICTIONS

results = X_test.copy()

results["Actual_Churn"] = y_test
results["Predicted_Churn"] = y_pred

results.to_csv(
    "outputs/customer_churn_predictions.csv",
    index=False
)

print("\nFirst 10 Prediction Results:")
print(results.head(10).to_string(index=False))


# 12. FINAL SUMMARY

print("\n" + "=" * 60)
print("PROJECT EXECUTION COMPLETED")
print("=" * 60)

print(f"Total Customers Analysed: {len(df)}")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

print("\nOutput files saved in the 'outputs' folder:")
print("1. Churn distribution graph")
print("2. Churn by geography graph")
print("3. Age distribution graph")
print("4. Churn by products graph")
print("5. Confusion matrix")
print("6. Feature importance graph")
print("7. Model evaluation results")
print("8. Customer churn predictions")

print("\nThank you!")

