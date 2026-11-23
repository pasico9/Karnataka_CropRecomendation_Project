"""
Karnataka Crop Project - Phase 3: Data Selection
KDD Stage: Data Selection (choose relevant columns/regions/years for each
downstream mining task, instead of feeding raw fact tables to the models)

Input  : data/warehouse/karnataka_crop_dw.db
Output : data/selection/selection_classification.csv  (-> crop recommendation model)
         data/selection/selection_clustering.csv       (-> district productivity clustering)
         data/selection/selection_trend.csv             (-> yield trend analysis over years)
"""

import pandas as pd
import sqlite3
import os

DB_PATH = "data/warehouse/karnataka_crop_dw.db"
OUT_DIR = "data/selection"
os.makedirs(OUT_DIR, exist_ok=True)

conn = sqlite3.connect(DB_PATH)


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# 1. CLASSIFICATION DATASET  ->  soil/climate + district -> recommended crop
# ---------------------------------------------------------------------------
# Decision: district_id IS included as a feature (region_name), since this
# project's focus is regional crop planning, not a generic soil/climate-only
# recommender. To test a district-free model instead, just drop the
# 'district_name' / 'region' columns before training -- everything else
# stays the same.

section("1. Classification dataset (crop recommendation)")

classification_query = """
SELECT
    f.N, f.P, f.K, f.temperature, f.humidity, f.ph, f.rainfall,
    s.soil_type,
    se.season_name,
    d.district_name,
    d.region,
    c.crop_name AS target_crop
FROM Fact_Crop_Recommendation f
JOIN Dim_Soil s     ON f.soil_id = s.soil_id
JOIN Dim_Season se  ON f.season_id = se.season_id
JOIN Dim_District d ON f.district_id = d.district_id
JOIN Dim_Crop c     ON f.crop_id = c.crop_id
"""
df_class = pd.read_sql(classification_query, conn)
df_class.to_csv(f"{OUT_DIR}/selection_classification.csv", index=False)

print("Shape:", df_class.shape)
print("Columns:", list(df_class.columns))
print("Target classes:", df_class["target_crop"].nunique())
print(f"Saved -> {OUT_DIR}/selection_classification.csv")


# ---------------------------------------------------------------------------
# 2. CLUSTERING DATASET  ->  district-level productivity profile
# ---------------------------------------------------------------------------
# Clustering raw yearly rows would group by year-to-year noise rather than
# by district character, so this aggregates Fact_Regional_Yield up to one
# row per district first. price is dropped per project decision (economic,
# not agronomic). dominant_crop / dominant_soil use the most frequent value
# per district as a categorical summary alongside the numeric averages.

section("2. Clustering dataset (district productivity profile)")

yield_detail = pd.read_sql("""
SELECT
    d.district_name, d.region,
    c.crop_name, s.soil_type,
    f.area, f.rainfall, f.temperature, f.humidity, f.yield
FROM Fact_Regional_Yield f
JOIN Dim_District d ON f.district_id = d.district_id
JOIN Dim_Crop c     ON f.crop_id = c.crop_id
JOIN Dim_Soil s     ON f.soil_id = s.soil_id
""", conn)

agg = yield_detail.groupby(["district_name", "region"]).agg(
    avg_yield=("yield", "mean"),
    avg_area=("area", "mean"),
    avg_rainfall=("rainfall", "mean"),
    avg_temperature=("temperature", "mean"),
    avg_humidity=("humidity", "mean"),
    record_count=("yield", "size"),
).reset_index()

dominant_crop = (
    yield_detail.groupby("district_name")["crop_name"]
    .agg(lambda x: x.mode().iloc[0])
    .rename("dominant_crop")
)
dominant_soil = (
    yield_detail.groupby("district_name")["soil_type"]
    .agg(lambda x: x.mode().iloc[0])
    .rename("dominant_soil")
)

df_cluster = agg.merge(dominant_crop, on="district_name").merge(dominant_soil, on="district_name")
df_cluster = df_cluster.round(2)
df_cluster.to_csv(f"{OUT_DIR}/selection_clustering.csv", index=False)

print("Shape:", df_cluster.shape, "(one row per district)")
print(df_cluster[["district_name", "avg_yield", "dominant_crop", "record_count"]])
print(f"Saved -> {OUT_DIR}/selection_clustering.csv")


# ---------------------------------------------------------------------------
# 3. TREND DATASET  ->  year x district x crop -> yield, for trend analysis
# ---------------------------------------------------------------------------
# Kept at full grain (not aggregated) since trend analysis needs the
# year-by-year movement, not a single summary number. price dropped here too.

section("3. Trend dataset (yield over years)")

trend_query = """
SELECT
    y.year, d.district_name, d.region, c.crop_name,
    f.area, f.yield, f.rainfall, f.temperature
FROM Fact_Regional_Yield f
JOIN Dim_Year y     ON f.year_id = y.year_id
JOIN Dim_District d ON f.district_id = d.district_id
JOIN Dim_Crop c     ON f.crop_id = c.crop_id
ORDER BY y.year, d.district_name, c.crop_name
"""
df_trend = pd.read_sql(trend_query, conn)
df_trend.to_csv(f"{OUT_DIR}/selection_trend.csv", index=False)

print("Shape:", df_trend.shape)
print("Year range:", df_trend["year"].min(), "-", df_trend["year"].max())
print(f"Saved -> {OUT_DIR}/selection_trend.csv")

conn.close()

section("DONE")
print("Three selection-stage datasets written to data/selection/.")
print("Proceed to Data Transformation next (scaling, encoding, feature engineering).")