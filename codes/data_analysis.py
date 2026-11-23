"""
Karnataka Crop Project - Phase 1: Data Cleaning
KDD Stage: Data Cleaning (missing values, duplicates, inconsistent names)

Input  : data/raw_data_1.csv   (soil/climate -> crop recommendation data)
         data/raw_data_2.csv   (year-wise regional production/yield data)
Output : data/cleaned/cleaned_data_1.csv
         data/cleaned/cleaned_data_2.csv
"""

import pandas as pd
import numpy as np
import os

RAW_DIR = "data"
OUT_DIR = "data/cleaned"
os.makedirs(OUT_DIR, exist_ok=True)

pd.set_option("display.width", 120)


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_text_col(series):
    """Trim whitespace, collapse internal multi-spaces, Title Case."""
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.title()
    )


def report_outliers(df, cols):
    """Print IQR-based outlier counts per numeric column (report only, no removal)."""
    for c in cols:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((df[c] < low) | (df[c] > high)).sum()
        print(f"  {c:15s} range=[{df[c].min():.2f}, {df[c].max():.2f}]  "
              f"IQR_outliers={n_out} ({n_out/len(df)*100:.1f}%)")


# ---------------------------------------------------------------------------
# FILE 1: Soil/Climate -> Crop recommendation dataset
# ---------------------------------------------------------------------------

section("FILE 1: raw_data_1.csv (crop recommendation)")

df1 = pd.read_csv(f"{RAW_DIR}/raw_data_1.csv")
print("Original shape:", df1.shape)

# 1. Duplicates
dupes = df1.duplicated().sum()
df1 = df1.drop_duplicates()
print(f"Exact duplicate rows removed: {dupes}")

# 2. Missing values
print("Missing values per column:\n", df1.isnull().sum().to_string())
# (dataset had 0 nulls at inspection time; drop any full-NaN rows defensively)
df1 = df1.dropna(how="all")

# 3. Standardize text/categorical columns
text_cols_1 = ["soil_type", "season", "district", "region", "label"]
for c in text_cols_1:
    df1[c] = clean_text_col(df1[c])

print("\nCategories after cleaning:")
for c in text_cols_1:
    print(f"  {c}: {df1[c].nunique()} unique -> {sorted(df1[c].unique())[:8]}...")

# 4. Outlier check on numeric columns (report only — these are natural
#    soil/climate readings, not errors, so we flag rather than delete)
numeric_cols_1 = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
print("\nOutlier report (IQR method):")
report_outliers(df1, numeric_cols_1)

# 5. Sanity bounds (domain knowledge: pH must be 0-14, humidity/rainfall etc. non-negative)
before = len(df1)
df1 = df1[(df1["ph"].between(0, 14)) & (df1[numeric_cols_1] >= 0).all(axis=1)]
print(f"\nRows dropped for impossible values (e.g. negative rainfall, ph outside 0-14): {before - len(df1)}")

# 6. Class balance check
print("\nCrop label distribution (min / max count):")
counts = df1["label"].value_counts()
print(f"  {counts.idxmin()}: {counts.min()}   |   {counts.idxmax()}: {counts.max()}")

print("\nFinal shape:", df1.shape)
df1.to_csv(f"{OUT_DIR}/cleaned_data_1.csv", index=False)
print(f"Saved -> {OUT_DIR}/cleaned_data_1.csv")


# ---------------------------------------------------------------------------
# FILE 2: Regional yield/production dataset
# ---------------------------------------------------------------------------

section("FILE 2: raw_data_2.csv (regional yield/production)")

df2 = pd.read_csv(f"{RAW_DIR}/raw_data_2.csv")
print("Original shape:", df2.shape)

# 1. Standardize column names (consistency with file 1 + fix typo)
df2 = df2.rename(columns={
    "Soil type": "soil_type",
    "yeilds": "yield",
    "Location": "location",
    "Crops": "crop",
    "Season": "season",
    "Irrigation": "irrigation",
    "Rainfall": "rainfall",
    "Temperature": "temperature",
    "Humidity": "humidity",
    "Area": "area",
    "Year": "year",
    "price": "price",
})

# 2. Duplicates
dupes = df2.duplicated().sum()
df2 = df2.drop_duplicates()
print(f"Exact duplicate rows removed: {dupes}")

# 3. Missing values
print("Missing values per column:\n", df2.isnull().sum().to_string())

# 4. Standardize text columns FIRST (so grouping for imputation works correctly)
text_cols_2 = ["soil_type", "location", "crop", "season", "irrigation"]
for c in text_cols_2:
    df2[c] = clean_text_col(df2[c].fillna("MISSING")).replace("Missing", np.nan)

# 5. Collapse near-duplicate soil_type categories into a clean vocabulary.
#    Categories that are genuinely ambiguous (drainage descriptors rather
#    than soil types, e.g. "Well Drained") are printed for manual review
#    instead of being silently guessed.
soil_map = {
    "Loam": "Loamy",
    "Loamy Sand": "Sandy Loam",
    "Black Cotton": "Black",
    "Heavy Cotton": "Black",
    "Sandy Loam": "Sandy Loam",
    "Clay Loam": "Clay Loam",
    "Sandy Clay Loam": "Sandy Clay Loam",
    "Medium Textured Clay": "Clay",
}
df2["soil_type"] = df2["soil_type"].replace(soil_map)

ambiguous = {"Well Drained", "Drained Loam", "Teelah", "Medium Textured",
             "River Basins", "Arid And Desert", "Light Sandy", "Dry Sandy",
             "Gravelly Sand", "Heavy Clay", "Red Laterite"}
found_ambiguous = set(df2["soil_type"].dropna().unique()) & ambiguous
if found_ambiguous:
    print(f"\n[REVIEW] Kept as-is, not auto-merged (confirm manually): {sorted(found_ambiguous)}")

# 6. Impute missing soil_type: mode within same (location, crop) group,
#    fall back to overall mode if the group itself has no known soil_type
before_na = df2["soil_type"].isnull().sum()

def impute_group_mode(group):
    mode = group.mode()
    return mode.iloc[0] if not mode.empty else np.nan

group_mode = df2.groupby(["location", "crop"])["soil_type"].transform(impute_group_mode)
df2["soil_type"] = df2["soil_type"].fillna(group_mode)
overall_mode = df2["soil_type"].mode().iloc[0]
df2["soil_type"] = df2["soil_type"].fillna(overall_mode)
print(f"soil_type missing values imputed: {before_na} "
      f"(group mode where possible, else overall mode '{overall_mode}')")

print("\nFinal soil_type categories:", sorted(df2["soil_type"].unique()))

# 7. Outlier check on numeric columns
numeric_cols_2 = ["area", "rainfall", "temperature", "yield", "humidity", "price"]
print("\nOutlier report (IQR method):")
report_outliers(df2, numeric_cols_2)

# 8. Sanity bounds
before = len(df2)
df2 = df2[(df2[numeric_cols_2] >= 0).all(axis=1)]
print(f"Rows dropped for impossible negative values: {before - len(df2)}")

# 8b. Domain-based sanity check: Karnataka temperatures realistically fall
#     roughly between 10-48 C. Values found up to 222 C are clear data-entry
#     errors (e.g. Gulbarga rows show 128, 129, 130... - a sequential typo
#     pattern, not real readings). These rows are dropped rather than
#     capped/imputed, since the error also throws doubt on that row's other
#     fields.
before = len(df2)
bad_temp = ~df2["temperature"].between(10, 48)
print(f"\nRows with impossible temperature (outside 10-48 C): {bad_temp.sum()}")
df2 = df2[~bad_temp]
print(f"Rows dropped for bad temperature values: {before - len(df2)}")

print("\nFinal shape:", df2.shape)
df2.to_csv(f"{OUT_DIR}/cleaned_data_2.csv", index=False)
print(f"Saved -> {OUT_DIR}/cleaned_data_2.csv")


# ---------------------------------------------------------------------------
# Cross-file consistency check (useful for the Data Integration stage next)
# ---------------------------------------------------------------------------

section("CROSS-FILE CHECK (for next stage: Data Integration)")
locs2 = set(df2["location"].unique())
districts1 = set(df1["district"].unique())
print("Locations in File 2 not found in File 1 districts (spelling mismatches?):")
print(" ", sorted(locs2 - districts1))

section("DONE")
print("Cleaned files written to data/cleaned/. Proceed to Data Integration next.")