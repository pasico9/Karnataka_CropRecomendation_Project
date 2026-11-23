"""
Karnataka Crop Project - Phase 4: Data Transformation
KDD Stage: Data Transformation (scaling, aggregation, feature creation, encoding)

Each output is transformed differently because each feeds a different
algorithm with different requirements:
  - Classification (Random Forest): doesn't need scaled numerics, but
    categoricals must be encoded, and the target needs a label encoding.
  - Clustering (K-Means): IS distance-based, so every feature must be
    scaled to comparable ranges, or high-magnitude columns (like avg_area
    in the thousands) will dominate the distance calculation.
  - Trend analysis: feeds charts/visualization, not a model, so it stays
    human-readable -- only feature engineering is added, no scaling.

Input  : data/selection/selection_classification.csv
         data/selection/selection_clustering.csv
         data/selection/selection_trend.csv
Output : data/transformed/transformed_classification.csv
         data/transformed/transformed_clustering.csv
         data/transformed/transformed_trend.csv
         data/transformed/crop_label_mapping.csv   (decode target_crop_encoded -> name)
         data/transformed/scaler_params.csv         (documents scaling applied, for the report)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

IN_DIR = "data/selection"
OUT_DIR = "data/transformed"
os.makedirs(OUT_DIR, exist_ok=True)


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# 1. CLASSIFICATION TRANSFORM
# ---------------------------------------------------------------------------

section("1. Transforming classification dataset")

df_class = pd.read_csv(f"{IN_DIR}/selection_classification.csv")

numeric_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
categorical_cols = ["soil_type", "season_name", "district_name", "region"]

# Scale numeric features (StandardScaler: mean=0, std=1). Random Forest
# itself doesn't need this, but scaling now keeps the dataset reusable if
# you later compare against a distance-based or gradient-based model
# (KNN, SVM, logistic regression) without redoing this step.
scaler_class = StandardScaler()
scaled_numeric = pd.DataFrame(
    scaler_class.fit_transform(df_class[numeric_cols]),
    columns=[f"{c}_scaled" for c in numeric_cols],
)

# One-hot encode all categoricals (nominal, not ordinal -- no natural order
# to soil types or district names, so one-hot avoids implying one)
encoded_categorical = pd.get_dummies(df_class[categorical_cols], prefix=categorical_cols)

# Label-encode the target (models need a single integer column, not text)
crop_encoder = LabelEncoder()
target_encoded = crop_encoder.fit_transform(df_class["target_crop"])

df_class_transformed = pd.concat(
    [scaled_numeric, encoded_categorical], axis=1
)
df_class_transformed["target_crop_encoded"] = target_encoded

df_class_transformed.to_csv(f"{OUT_DIR}/transformed_classification.csv", index=False)

# Save the label mapping so predictions can be decoded back to crop names later
crop_mapping = pd.DataFrame({
    "target_crop_encoded": range(len(crop_encoder.classes_)),
    "crop_name": crop_encoder.classes_,
})
crop_mapping.to_csv(f"{OUT_DIR}/crop_label_mapping.csv", index=False)

print("Shape:", df_class_transformed.shape)
print(f"  {len(numeric_cols)} numeric features scaled")
print(f"  {len(categorical_cols)} categoricals one-hot encoded -> {encoded_categorical.shape[1]} dummy columns")
n_classes = len(crop_encoder.classes_)
print(f"  Target: {n_classes} crops label-encoded 0-{n_classes - 1}")
print(f"Saved -> {OUT_DIR}/transformed_classification.csv")
print(f"Saved -> {OUT_DIR}/crop_label_mapping.csv")


# ---------------------------------------------------------------------------
# 2. CLUSTERING TRANSFORM
# ---------------------------------------------------------------------------

section("2. Transforming clustering dataset")

df_cluster = pd.read_csv(f"{IN_DIR}/selection_clustering.csv")

# Feature engineering: yield per unit area is a fairer productivity signal
# than raw avg_yield, since districts differ hugely in cultivated area.
df_cluster["yield_per_area"] = df_cluster["avg_yield"] / df_cluster["avg_area"]

# record_count is a sampling artifact of the source data (how many rows
# exist), not a real district characteristic -- excluded as a clustering
# feature. dominant_crop/dominant_soil are also excluded (both came out
# uninformative -- "Coconut" for every district, see Phase 3 notes) but
# kept as reference/label columns for interpreting clusters afterward.
cluster_feature_cols = ["avg_yield", "avg_area", "avg_rainfall",
                         "avg_temperature", "avg_humidity", "yield_per_area"]

scaler_cluster = StandardScaler()
scaled_features = pd.DataFrame(
    scaler_cluster.fit_transform(df_cluster[cluster_feature_cols]),
    columns=[f"{c}_scaled" for c in cluster_feature_cols],
)

df_cluster_transformed = pd.concat([
    df_cluster[["district_name", "region", "dominant_crop", "dominant_soil"]],
    scaled_features,
], axis=1)

df_cluster_transformed.to_csv(f"{OUT_DIR}/transformed_clustering.csv", index=False)

print("Shape:", df_cluster_transformed.shape)
print(f"  Features scaled for K-Means: {cluster_feature_cols}")
print("  district_name/region/dominant_crop/dominant_soil kept as labels only (not clustering inputs)")
print(f"Saved -> {OUT_DIR}/transformed_clustering.csv")


# ---------------------------------------------------------------------------
# 3. TREND TRANSFORM
# ---------------------------------------------------------------------------

section("3. Transforming trend dataset")

df_trend = pd.read_csv(f"{IN_DIR}/selection_trend.csv")

# Same productivity feature as above, at per-record grain this time, so
# districts/crops with very different cultivated areas can be compared
# fairly on charts (raw yield alone favors large-area records).
df_trend["yield_per_area"] = df_trend["yield"] / df_trend["area"]

# No scaling here: this dataset feeds line/bar charts for the dashboard,
# not a distance-based model, so real units (kept human-readable) are
# more useful than standardized scores.
df_trend.to_csv(f"{OUT_DIR}/transformed_trend.csv", index=False)

print("Shape:", df_trend.shape)
print("  Added yield_per_area feature; no scaling applied (feeds visualization, not a model)")
print(f"Saved -> {OUT_DIR}/transformed_trend.csv")


# ---------------------------------------------------------------------------
# Document the scaling applied (useful for your project report)
# ---------------------------------------------------------------------------

section("Saving scaler documentation")

scaler_doc = pd.DataFrame({
    "dataset": ["classification"] * len(numeric_cols) + ["clustering"] * len(cluster_feature_cols),
    "feature": numeric_cols + cluster_feature_cols,
    "mean": list(scaler_class.mean_) + list(scaler_cluster.mean_),
    "std": list(scaler_class.scale_) + list(scaler_cluster.scale_),
})
scaler_doc.to_csv(f"{OUT_DIR}/scaler_params.csv", index=False)
print(f"Saved -> {OUT_DIR}/scaler_params.csv")

section("DONE")
print("All three transformed datasets are model-ready.")
print("Proceed to Data Mining next (Random Forest classifier + K-Means clustering).")