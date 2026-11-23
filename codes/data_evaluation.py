"""
Karnataka Crop Project - Phase 6: Pattern Evaluation
KDD Stage: Pattern Evaluation (accuracy, F1, confusion matrix, silhouette score)

Two things get proper scrutiny here that Phase 5 only flagged:
  1. The Random Forest had 100% train accuracy in Phase 5 -- a classic
     overfitting signal. This script cross-validates across max_depth
     values to check whether a shallower, more honest model performs
     just as well on unseen data.
  2. k=3 for K-Means was chosen from the elbow curve alone. This script
     computes the silhouette score for each candidate k to check whether
     k=3 is actually well-separated, not just "the bend in the curve".

Input  : data/transformed/transformed_classification.csv
         data/transformed/transformed_clustering.csv
         data/transformed/crop_label_mapping.csv
Output : data/evaluation/classification_report.csv
         data/evaluation/confusion_matrix.csv
         data/evaluation/depth_tuning_results.csv
         data/evaluation/rf_crop_model_tuned.pkl        (replaces Phase 5's model if better)
         data/evaluation/silhouette_scores.csv
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, classification_report,
    confusion_matrix, silhouette_score,
)
from sklearn.cluster import KMeans

OUT_DIR = "data/evaluation"
os.makedirs(OUT_DIR, exist_ok=True)


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# 1. CLASSIFICATION: full metrics on the Phase 5 model
# ---------------------------------------------------------------------------

section("1. Random Forest - full evaluation")

df_class = pd.read_csv("data/transformed/transformed_classification.csv")
crop_mapping = pd.read_csv("data/transformed/crop_label_mapping.csv")
crop_lookup = dict(zip(crop_mapping["target_crop_encoded"], crop_mapping["crop_name"]))

X = df_class.drop(columns=["target_crop_encoded"])
y = df_class["target_crop_encoded"]

# Same split as Phase 5 (same random_state) so results are directly comparable
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

baseline_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
baseline_model.fit(X_train, y_train)
y_pred_baseline = baseline_model.predict(X_test)

print(f"Baseline (unrestricted depth) -- train acc: {baseline_model.score(X_train, y_train):.3f}, "
      f"test acc: {accuracy_score(y_test, y_pred_baseline):.3f}")
print("Gap between train and test accuracy is the overfitting signal from Phase 5.")

# ---- Address the overfitting flag: cross-validate across max_depth ----
section("1b. Depth tuning to check for overfitting")

depth_options = [5, 10, 15, 20, 30, None]
tuning_results = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for depth in depth_options:
    model = RandomForestClassifier(n_estimators=200, max_depth=depth, random_state=42, n_jobs=-1)
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="accuracy", n_jobs=-1)
    tuning_results.append({
        "max_depth": depth if depth is not None else "None (unlimited)",
        "cv_mean_accuracy": cv_scores.mean(),
        "cv_std": cv_scores.std(),
    })
    print(f"  max_depth={str(depth):>18s}: CV accuracy = {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

tuning_df = pd.DataFrame(tuning_results)
tuning_df.to_csv(f"{OUT_DIR}/depth_tuning_results.csv", index=False)

# Pick the shallowest depth within 1 std of the best CV score (simpler
# model, similar performance -- avoids picking unlimited depth just
# because it's marginally higher on a small CV difference)
best_idx = tuning_df["cv_mean_accuracy"].idxmax()
best_score = tuning_df.loc[best_idx, "cv_mean_accuracy"]
best_std = tuning_df.loc[best_idx, "cv_std"]
within_1std = tuning_df[tuning_df["cv_mean_accuracy"] >= best_score - best_std]
chosen_depth_label = within_1std.iloc[0]["max_depth"]
chosen_depth = None if chosen_depth_label == "None (unlimited)" else int(chosen_depth_label)

print(f"\nChosen max_depth = {chosen_depth_label} (simplest model within 1 std of best CV score)")

final_model = RandomForestClassifier(n_estimators=200, max_depth=chosen_depth, random_state=42, n_jobs=-1)
final_model.fit(X_train, y_train)
y_pred_final = final_model.predict(X_test)

final_train_acc = final_model.score(X_train, y_train)
final_test_acc = accuracy_score(y_test, y_pred_final)
print(f"Tuned model -- train acc: {final_train_acc:.3f}, test acc: {final_test_acc:.3f} "
      f"(gap: {final_train_acc - final_test_acc:.3f}, vs baseline gap: "
      f"{baseline_model.score(X_train, y_train) - accuracy_score(y_test, y_pred_baseline):.3f})")

# ---- Full metrics on the tuned model ----
section("1c. Full classification metrics (tuned model)")

f1_macro = f1_score(y_test, y_pred_final, average="macro")
f1_weighted = f1_score(y_test, y_pred_final, average="weighted")
precision_weighted = precision_score(y_test, y_pred_final, average="weighted", zero_division=0)
print(f"F1 (macro):    {f1_macro:.3f}   -- treats every crop class equally regardless of frequency")
print(f"F1 (weighted): {f1_weighted:.3f}   -- weighted by how common each crop is in the test set")
print(f"Precision (weighted): {precision_weighted:.3f}")

present_labels = sorted(y_test.unique())
report_dict = classification_report(
    y_test, y_pred_final,
    labels=present_labels,
    target_names=[crop_lookup[i] for i in present_labels],
    output_dict=True, zero_division=0,
)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv(f"{OUT_DIR}/classification_report.csv")
print("\nPer-class report saved. Weakest classes (lowest F1):")
worst = report_df.drop(["accuracy", "macro avg", "weighted avg"], errors="ignore").sort_values("f1-score").head(5)
print(worst[["precision", "recall", "f1-score", "support"]].to_string())

cm = confusion_matrix(y_test, y_pred_final, labels=present_labels)
cm_df = pd.DataFrame(
    cm,
    index=[crop_lookup[i] for i in present_labels],
    columns=[crop_lookup[i] for i in present_labels],
)
cm_df.to_csv(f"{OUT_DIR}/confusion_matrix.csv")
print(f"\nSaved -> {OUT_DIR}/classification_report.csv")
print(f"Saved -> {OUT_DIR}/confusion_matrix.csv")

joblib.dump(final_model, f"{OUT_DIR}/rf_crop_model_tuned.pkl")
print(f"Saved tuned model -> {OUT_DIR}/rf_crop_model_tuned.pkl (use this one going forward, not Phase 5's)")

# Save headline metrics for the dashboard to read at runtime, instead of
# ever hard-coding a number like "91.8%" in app.py -- if this script is
# rerun and numbers change, the dashboard picks up the new values
# automatically on its next load.
metrics_summary = pd.DataFrame([{
    "test_accuracy": final_test_acc,
    "f1_macro": f1_macro,
    "f1_weighted": f1_weighted,
    "precision_weighted": precision_weighted,
    "chosen_max_depth": chosen_depth_label,
    "baseline_train_accuracy": baseline_model.score(X_train, y_train),
    "baseline_test_accuracy": accuracy_score(y_test, y_pred_baseline),
    "tuned_train_accuracy": final_train_acc,
    "tuned_test_accuracy": final_test_acc,
}])
metrics_summary.to_csv(f"{OUT_DIR}/evaluation_metrics.csv", index=False)
print(f"Saved -> {OUT_DIR}/evaluation_metrics.csv (dashboard reads this instead of hard-coded numbers)")


# ---------------------------------------------------------------------------
# 2. CLUSTERING: silhouette score to validate k=3
# ---------------------------------------------------------------------------

section("2. K-Means - silhouette analysis")

df_cluster = pd.read_csv("data/transformed/transformed_clustering.csv")
feature_cols = [c for c in df_cluster.columns if c.endswith("_scaled")]
X_cluster = df_cluster[feature_cols]

sil_results = []
# silhouette_score needs 2 <= k <= n_samples - 1; only 9 districts here
for k in range(2, min(6, len(X_cluster) - 1)):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_cluster)
    score = silhouette_score(X_cluster, labels)
    sil_results.append({"k": k, "silhouette_score": score})
    print(f"  k={k}: silhouette = {score:.3f}")

sil_df = pd.DataFrame(sil_results)
sil_df.to_csv(f"{OUT_DIR}/silhouette_scores.csv", index=False)

best_k = sil_df.loc[sil_df["silhouette_score"].idxmax(), "k"]
print(f"\nBest silhouette score is at k={int(best_k)}.")
if best_k != 3:
    print(f"NOTE: this differs from the k=3 chosen by the elbow method in Phase 5 -- "
          f"with only 9 data points, both methods are weak signals. Worth reporting "
          f"both and being upfront about the small sample size in your write-up.")
else:
    print("This confirms k=3 from the elbow method in Phase 5.")

print(f"\nSaved -> {OUT_DIR}/silhouette_scores.csv")


section("DONE")
print("Evaluation complete. Use rf_crop_model_tuned.pkl (not Phase 5's model) for the dashboard.")
print("Proceed to Knowledge Presentation next (dashboard, charts, recommendations).")