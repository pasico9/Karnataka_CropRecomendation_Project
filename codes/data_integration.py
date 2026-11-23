"""
Karnataka Crop Project - Phase 2: Data Integration
KDD Stage: Data Integration (combine crop + regional + soil/climate datasets
into a consistent, conformed structure ready for warehouse loading)

Input  : data/cleaned/cleaned_data_1.csv   (crop recommendation)
         data/cleaned/cleaned_data_2.csv   (regional yield/production)
Output : data/warehouse/karnataka_crop_dw.db   (SQLite star schema)
         data/warehouse/*.csv               (each dim/fact table, for inspection)
"""

import pandas as pd
import numpy as np
import sqlite3
import os

CLEAN_DIR = "data/cleaned"
OUT_DIR = "data/warehouse"
DB_PATH = f"{OUT_DIR}/karnataka_crop_dw.db"
os.makedirs(OUT_DIR, exist_ok=True)

if os.path.exists(DB_PATH):
    os.remove(DB_PATH)  # rebuild fresh each run


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


df1 = pd.read_csv(f"{CLEAN_DIR}/cleaned_data_1.csv")   # crop recommendation
df2 = pd.read_csv(f"{CLEAN_DIR}/cleaned_data_2.csv")   # regional yield


# ---------------------------------------------------------------------------
# Conformance maps: reconcile naming differences between the two sources
# ---------------------------------------------------------------------------

# File 2's 'location' -> canonical district name used in File 1.
# Madikeri is the headquarters town of Kodagu district, so both
# 'Kodagu' and 'Madikeri' rows correctly map to the same district.
# Kasaragodu is in Kerala, not Karnataka -- kept as its own entry
# per project decision, tagged with a distinct region.
DISTRICT_MAP = {
    "Chikmangaluru": "Chikkamagaluru",
    "Davangere": "Davanagere",
    "Gulbarga": "Kalaburagi",
    "Hassan": "Hassan",
    "Kasaragodu": "Kasaragodu (Kerala)",
    "Kodagu": "Kodagu",
    "Madikeri": "Kodagu",
    "Mangalore": "Dakshina Kannada",
    "Mysuru": "Mysuru",
    "Raichur": "Raichur",
}

# File 2's crop name -> canonical crop name used in File 1
CROP_MAP = {
    "Cardamum": "Cardamom",
    "Paddy": "Rice",
}


# ---------------------------------------------------------------------------
# DIM_DISTRICT  (district_id, district_name, region)
# ---------------------------------------------------------------------------

section("Building Dim_District")

dim_district = (
    df1[["district", "region"]]
    .drop_duplicates()
    .rename(columns={"district": "district_name"})
)
# add Kasaragodu (Kerala) as its own district/region row
dim_district = pd.concat([
    dim_district,
    pd.DataFrame([{"district_name": "Kasaragodu (Kerala)", "region": "Other (Kerala)"}])
], ignore_index=True)

dim_district = dim_district.sort_values("district_name").reset_index(drop=True)
dim_district.insert(0, "district_id", range(1, len(dim_district) + 1))
print(dim_district)

district_lookup = dict(zip(dim_district["district_name"], dim_district["district_id"]))


# ---------------------------------------------------------------------------
# DIM_CROP  (crop_id, crop_name)
# ---------------------------------------------------------------------------

section("Building Dim_Crop")

crops_1 = set(df1["label"].unique())
crops_2_mapped = set(df2["crop"].map(lambda c: CROP_MAP.get(c, c)).unique())
all_crops = sorted(crops_1 | crops_2_mapped)

dim_crop = pd.DataFrame({"crop_name": all_crops})
dim_crop.insert(0, "crop_id", range(1, len(dim_crop) + 1))
print(f"{len(dim_crop)} unique crops (File 1: {len(crops_1)}, File 2 mapped: {len(crops_2_mapped)}, "
      f"overlap: {len(crops_1 & crops_2_mapped)})")

crop_lookup = dict(zip(dim_crop["crop_name"], dim_crop["crop_id"]))


# ---------------------------------------------------------------------------
# DIM_SEASON  (season_id, season_name)
# ---------------------------------------------------------------------------

section("Building Dim_Season")

all_seasons = sorted(set(df1["season"].unique()) | set(df2["season"].unique()))
dim_season = pd.DataFrame({"season_name": all_seasons})
dim_season.insert(0, "season_id", range(1, len(dim_season) + 1))
print(dim_season)

season_lookup = dict(zip(dim_season["season_name"], dim_season["season_id"]))


# ---------------------------------------------------------------------------
# DIM_SOIL  (soil_id, soil_type)
# ---------------------------------------------------------------------------

section("Building Dim_Soil")

all_soils = sorted(set(df1["soil_type"].unique()) | set(df2["soil_type"].unique()))
dim_soil = pd.DataFrame({"soil_type": all_soils})
dim_soil.insert(0, "soil_id", range(1, len(dim_soil) + 1))
print(f"{len(dim_soil)} unique soil types (union of both sources)")

soil_lookup = dict(zip(dim_soil["soil_type"], dim_soil["soil_id"]))


# ---------------------------------------------------------------------------
# DIM_YEAR  (year_id, year)  -- only File 2 has years
# ---------------------------------------------------------------------------

section("Building Dim_Year")

all_years = sorted(df2["year"].unique())
dim_year = pd.DataFrame({"year": all_years})
dim_year.insert(0, "year_id", range(1, len(dim_year) + 1))
print(dim_year)

year_lookup = dict(zip(dim_year["year"], dim_year["year_id"]))


# ---------------------------------------------------------------------------
# FACT_CROP_RECOMMENDATION
# ---------------------------------------------------------------------------

section("Building Fact_Crop_Recommendation")

fact_reco = df1.copy()
fact_reco["district_id"] = fact_reco["district"].map(district_lookup)
fact_reco["crop_id"] = fact_reco["label"].map(crop_lookup)
fact_reco["season_id"] = fact_reco["season"].map(season_lookup)
fact_reco["soil_id"] = fact_reco["soil_type"].map(soil_lookup)

fact_reco = fact_reco[[
    "district_id", "crop_id", "season_id", "soil_id",
    "N", "P", "K", "temperature", "humidity", "ph", "rainfall",
]].reset_index(drop=True)
fact_reco.insert(0, "reco_id", range(1, len(fact_reco) + 1))

orphans = fact_reco[["district_id", "crop_id", "season_id", "soil_id"]].isnull().sum()
print("Unmapped (orphan) foreign keys:\n", orphans.to_string())
print("Final shape:", fact_reco.shape)


# ---------------------------------------------------------------------------
# FACT_REGIONAL_YIELD
# ---------------------------------------------------------------------------

section("Building Fact_Regional_Yield")

fact_yield = df2.copy()
fact_yield["district_name"] = fact_yield["location"].map(DISTRICT_MAP)
fact_yield["crop_name"] = fact_yield["crop"].map(lambda c: CROP_MAP.get(c, c))

fact_yield["district_id"] = fact_yield["district_name"].map(district_lookup)
fact_yield["crop_id"] = fact_yield["crop_name"].map(crop_lookup)
fact_yield["season_id"] = fact_yield["season"].map(season_lookup)
fact_yield["soil_id"] = fact_yield["soil_type"].map(soil_lookup)
fact_yield["year_id"] = fact_yield["year"].map(year_lookup)

fact_yield = fact_yield[[
    "district_id", "crop_id", "season_id", "soil_id", "year_id",
    "irrigation", "area", "rainfall", "temperature", "humidity",
    "yield", "price",
]].reset_index(drop=True)
fact_yield.insert(0, "yield_id", range(1, len(fact_yield) + 1))

orphans = fact_yield[["district_id", "crop_id", "season_id", "soil_id", "year_id"]].isnull().sum()
print("Unmapped (orphan) foreign keys:\n", orphans.to_string())
print("Final shape:", fact_yield.shape)


# ---------------------------------------------------------------------------
# Save everything: SQLite DB (for querying/joining) + CSVs (for inspection)
# ---------------------------------------------------------------------------

section("Saving to SQLite warehouse + CSV exports")

conn = sqlite3.connect(DB_PATH)

tables = {
    "Dim_District": dim_district,
    "Dim_Crop": dim_crop,
    "Dim_Season": dim_season,
    "Dim_Soil": dim_soil,
    "Dim_Year": dim_year,
    "Fact_Crop_Recommendation": fact_reco,
    "Fact_Regional_Yield": fact_yield,
}

for name, tdf in tables.items():
    tdf.to_sql(name, conn, if_exists="replace", index=False)
    tdf.to_csv(f"{OUT_DIR}/{name}.csv", index=False)
    print(f"  {name:28s} {tdf.shape[0]:6d} rows  -> saved")

# Foreign key constraints (SQLite: informational unless PRAGMA foreign_keys=ON,
# but useful documentation + enables integrity checks by tools that read them)
conn.executescript("""
CREATE INDEX IF NOT EXISTS idx_reco_district ON Fact_Crop_Recommendation(district_id);
CREATE INDEX IF NOT EXISTS idx_reco_crop ON Fact_Crop_Recommendation(crop_id);
CREATE INDEX IF NOT EXISTS idx_yield_district ON Fact_Regional_Yield(district_id);
CREATE INDEX IF NOT EXISTS idx_yield_crop ON Fact_Regional_Yield(crop_id);
CREATE INDEX IF NOT EXISTS idx_yield_year ON Fact_Regional_Yield(year_id);
""")
conn.commit()

# Quick sanity join to prove the integration actually works end-to-end
section("Sanity check: sample joined query")
sample = pd.read_sql("""
    SELECT d.district_name, d.region, c.crop_name, y.year, f.yield, f.price
    FROM Fact_Regional_Yield f
    JOIN Dim_District d ON f.district_id = d.district_id
    JOIN Dim_Crop c ON f.crop_id = c.crop_id
    JOIN Dim_Year y ON f.year_id = y.year_id
    WHERE d.region = 'Other (Kerala)'
    LIMIT 5
""", conn)
print(sample)

conn.close()
print(f"\nWarehouse saved -> {DB_PATH}")
