import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os

print("Starting data preparation for fully associative model...")

# --- Feature Association Logic ---


def create_lookup_maps(df: pd.DataFrame):
    """Generates maps for market and commodity associations, and overall defaults."""

    # 1. Market Lookup Map (Location Details: admin1, admin2, lat, lon)
    market_lookup_df = df.groupby('market').agg(
        admin1=('admin1', lambda x: x.mode()[0]),
        admin2=('admin2', lambda x: x.mode()[0]),
        latitude=('latitude', 'median'),
        longitude=('longitude', 'median')
    ).reset_index()
    market_lookup_map = market_lookup_df.set_index('market').to_dict('index')

    # 2. Commodity Lookup Map (Classification Details: category, unit, pricetype)
    commodity_lookup_df = df.groupby('commodity').agg(
        category=('category', lambda x: x.mode()[0]),
        unit=('unit', lambda x: x.mode()[0]),
        pricetype=('pricetype', lambda x: x.mode()[0]),
    ).reset_index()
    commodity_lookup_map = commodity_lookup_df.set_index(
        'commodity').to_dict('index')

    # 3. Overall Global Defaults (Fallback if both market/commodity are unknown)
    overall_defaults = {
        'admin1': df['admin1'].mode()[0],
        'admin2': df['admin2'].mode()[0],
        'latitude': df['latitude'].median(),
        'longitude': df['longitude'].median(),
        'category': df['category'].mode()[0],
        'unit': df['unit'].mode()[0],
        'pricetype': df['pricetype'].mode()[0],
    }

    return market_lookup_map, commodity_lookup_map, overall_defaults


# --- Load Data, Create Maps, and Clean ---
CSV_PATH = "dataset/wfp_food_prices_som.csv"
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=['latitude', 'longitude'])

# Create and save the lookup maps
MARKET_MAP, COMMODITY_MAP, OVERALL_DEFAULTS = create_lookup_maps(df.copy())
os.makedirs("models", exist_ok=True)
with open("models/market_lookup_maps.json", "w") as f:
    json.dump(MARKET_MAP, f, indent=4)
with open("models/commodity_lookup_maps.json", "w") as f:
    json.dump(COMMODITY_MAP, f, indent=4)
with open("models/overall_defaults.json", "w") as f:
    json.dump(OVERALL_DEFAULTS, f, indent=4)

# (Standard preprocessing continues below)
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df = df.drop(columns=["usdprice", "priceflag",
             "currency", "date"], errors="ignore")

# Impute and Outlier handling (standard)
categorical_cols = ["admin1", "admin2", "market",
                    "category", "commodity", "unit", "pricetype"]
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])


def iqr_bounds(series, k=1.5):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return lower, upper


low_price, high_price = iqr_bounds(df["price"])
df["price"] = df["price"].clip(lower=low_price, upper=high_price)


# --- ADAPTIVE OHE: Grouping Rare Categories (for robustness) ---
KNOWN_CATEGORIES = {}
RARE_THRESHOLD = 100
for col in categorical_cols:
    value_counts = df[col].value_counts()
    known_values = value_counts[value_counts >= RARE_THRESHOLD].index.tolist()
    KNOWN_CATEGORIES[col] = known_values
    df[col] = np.where(df[col].isin(known_values), df[col], 'OTHER')
with open("models/known_categories.json", "w") as f:
    json.dump(KNOWN_CATEGORIES, f, indent=4)

# --- Final Encoding and Scaling ---
df = pd.get_dummies(df, columns=categorical_cols, drop_first=False, dtype=int)
df["price_per_lat"] = 1.0 / (df["latitude"].abs() + 1e-6)
df["price_per_long"] = 1.0 / (df["longitude"].abs() + 1e-6)
df["log_price"] = np.log1p(df["price"])

numeric_to_scale = ["latitude", "longitude", "year",
                    "month", "price_per_lat", "price_per_long"]
scaler = StandardScaler()
df[numeric_to_scale] = scaler.fit_transform(df[numeric_to_scale])

joblib.dump(scaler, "models/food_scaler.pkl")
TRAIN_COLUMNS = df.drop(columns=["price", "log_price"]).columns.tolist()
json.dump(TRAIN_COLUMNS, open("models/train_columns.json", "w"))

df.to_csv("dataset/clean_food_prices_som.csv", index=False)

print("\n✅ Preprocessing finished. Cascading feature association maps saved.")
