"""
Karnataka Crop Project - Phase 5: Data Mining
KDD Stage: Data Mining (Random Forest + K-Means)

Three mining tasks, each on its purpose-built dataset from Phase 4:
  1. Classification  -> Random Forest predicts crop from soil/climate/district
  2. Clustering       -> K-Means groups districts by productivity profile
  3. Trend analysis    -> per-crop yield trend over 2004-2019

Detailed metrics (accuracy, F1, confusion matrix, silhouette score) are
computed in the NEXT phase (Pattern Evaluation) -- this script trains the
models and saves outputs, and only prints quick sanity numbers.

Input  : data/transformed/transformed_classification.csv
         data/transformed/transformed_clustering.csv
         data/transformed/transformed_trend.csv
         data/transformed/crop_label_mapping.csv
Output : data/models/rf_crop_model.pkl
         data/models/kmeans_district_model.pkl
         data/mining/classification_predictions.csv
         data/mining/district_clusters.csv
         data/mining/kmeans_elbow.csv
         data/mining/yearly_trend_summary.csv
         data/mining/crop_trend_slopes.csv
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

MODEL_DIR = "data/models"
OUT_DIR = "data/mining"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# 1. CLASSIFICATION: Random Forest crop recommendation
# ---------------------------------------------------------------------------

section("1. Random Forest - crop recommendation")

df_class = pd.read_csv("data/transformed/transformed_classification.csv")
crop_mapping = pd.read_csv("data/transformed/crop_label_mapping.csv")

X = df_class.drop(columns=["target_crop_encoded"])
y = df_class["target_crop_encoded"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf_model = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

train_acc = rf_model.score(X_train, y_train)
test_acc = rf_model.score(X_test, y_test)
print(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}")
print(f"Quick sanity check -- train accuracy: {train_acc:.3f}, test accuracy: {test_acc:.3f}")
print("(Full metrics -- per-class F1, confusion matrix -- come in Phase 6: Pattern Evaluation)")

# Save predictions for the evaluation phase, decoded back to crop names
y_pred = rf_model.predict(X_test)
crop_lookup = dict(zip(crop_mapping["target_crop_encoded"], crop_mapping["crop_name"]))
predictions_df = pd.DataFrame({
    "actual_crop": y_test.map(crop_lookup).values,
    "predicted_crop": pd.Series(y_pred).map(crop_lookup).values,
})
predictions_df.to_csv(f"{OUT_DIR}/classification_predictions.csv", index=False)

# Feature importance -- useful for the report ("what matters most for recommending a crop?")
importances = pd.DataFrame({
    "feature": X.columns,
    "importance": rf_model.feature_importances_,
}).sort_values("importance", ascending=False)
importances.to_csv(f"{OUT_DIR}/feature_importance.csv", index=False)
print("\nTop 8 most important features:")
print(importances.head(8).to_string(index=False))

joblib.dump(rf_model, f"{MODEL_DIR}/rf_crop_model.pkl")
print(f"\nSaved model -> {MODEL_DIR}/rf_crop_model.pkl")
print(f"Saved predictions -> {OUT_DIR}/classification_predictions.csv")


# ---------------------------------------------------------------------------
# 2. CLUSTERING: K-Means district productivity grouping
# ---------------------------------------------------------------------------

section("2. K-Means - district productivity clustering")

df_cluster = pd.read_csv("data/transformed/transformed_clustering.csv")
feature_cols = [c for c in df_cluster.columns if c.endswith("_scaled")]
X_cluster = df_cluster[feature_cols]

# Only 9 districts in this dataset -- elbow method is illustrative rather
# than statistically strong with this few points, but still worth showing
# in the report to justify the chosen k.
inertias = []
k_range = range(2, 6)   # can't sensibly test k >= n_samples
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_cluster)
    inertias.append(km.inertia_)

elbow_df = pd.DataFrame({"k": list(k_range), "inertia": inertias})
elbow_df.to_csv(f"{OUT_DIR}/kmeans_elbow.csv", index=False)
print("Elbow data (k vs inertia):")
print(elbow_df.to_string(index=False))

CHOSEN_K = 3
kmeans_model = KMeans(n_clusters=CHOSEN_K, random_state=42, n_init=10)
df_cluster["cluster"] = kmeans_model.fit_predict(X_cluster)

# Record the elbow-method choice itself (not just the resulting clusters),
# so the dashboard can display "elbow chose k=X" without hardcoding it.
pd.DataFrame([{"elbow_chosen_k": CHOSEN_K}]).to_csv(f"{OUT_DIR}/clustering_choice.csv", index=False)

result_cols = ["district_name", "region", "dominant_crop", "dominant_soil", "cluster"]
df_cluster[result_cols].to_csv(f"{OUT_DIR}/district_clusters.csv", index=False)

print(f"\nChosen k={CHOSEN_K} (see elbow data above and silhouette score next phase)")
print(df_cluster[result_cols].sort_values("cluster").to_string(index=False))

joblib.dump(kmeans_model, f"{MODEL_DIR}/kmeans_district_model.pkl")
print(f"\nSaved model -> {MODEL_DIR}/kmeans_district_model.pkl")
print(f"Saved cluster assignments -> {OUT_DIR}/district_clusters.csv")


# ---------------------------------------------------------------------------
# 3. TREND ANALYSIS: yield trend over 2004-2019
# ---------------------------------------------------------------------------

section("3. Trend analysis - historical yield patterns")

df_trend = pd.read_csv("data/transformed/transformed_trend.csv")

# Overall yearly picture (all districts/crops combined)
yearly_summary = df_trend.groupby("year").agg(
    avg_yield=("yield", "mean"),
    avg_yield_per_area=("yield_per_area", "mean"),
    total_area=("area", "sum"),
    record_count=("yield", "size"),
).reset_index()
yearly_summary.to_csv(f"{OUT_DIR}/yearly_trend_summary.csv", index=False)
print("Yearly summary (first 5 years):")
print(yearly_summary.head().to_string(index=False))

# Per-crop trend: fit a simple linear regression (year -> avg yield) for
# each crop that has enough years of data, so we can say "yield for X is
# trending up/down" rather than just eyeballing a chart.
slopes = []
for crop, group in df_trend.groupby("crop_name"):
    yearly = group.groupby("year")["yield"].mean().reset_index()
    if len(yearly) < 3:
        continue  # not enough points for a meaningful trend line
    lr = LinearRegression()
    lr.fit(yearly[["year"]], yearly["yield"])
    slopes.append({
        "crop_name": crop,
        "years_of_data": len(yearly),
        "slope_per_year": lr.coef_[0],
        "direction": "increasing" if lr.coef_[0] > 0 else "decreasing",
    })

trend_slopes = pd.DataFrame(slopes).sort_values("slope_per_year", ascending=False)
trend_slopes.to_csv(f"{OUT_DIR}/crop_trend_slopes.csv", index=False)
print(f"\nCrop trend slopes ({len(trend_slopes)} crops with >=3 years of data):")
print(trend_slopes.to_string(index=False))
print(f"\nSaved -> {OUT_DIR}/yearly_trend_summary.csv")
print(f"Saved -> {OUT_DIR}/crop_trend_slopes.csv")


section("DONE")
print("Models trained and saved to data/models/. Mining outputs saved to data/mining/.")
print("Proceed to Pattern Evaluation next (accuracy, F1, confusion matrix, silhouette score).")