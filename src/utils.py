import json
import joblib
import pandas as pd
import numpy as np

# Load assets
try:
    TRAIN_COLUMNS = json.load(open("models/train_columns.json"))
    SCALER = joblib.load("models/food_scaler.pkl")

    # LOAD NEW LOOKUP MAPS
    MARKET_LOOKUP_MAP = json.load(open("models/market_lookup_maps.json"))
    COMMODITY_LOOKUP_MAP = json.load(open("models/commodity_lookup_maps.json"))
    OVERALL_DEFAULTS = json.load(open("models/overall_defaults.json"))
    KNOWN_CATEGORIES = json.load(open("models/known_categories.json"))

except FileNotFoundError as e:
    print(
        f"ERROR: Model assets missing. Run preprocessing.py and model.py first. Details: {e}")
    TRAIN_COLUMNS, SCALER = [], object()
    MARKET_LOOKUP_MAP, COMMODITY_LOOKUP_MAP, OVERALL_DEFAULTS, KNOWN_CATEGORIES = {}, {}, {}, {}


def prepare_features_from_raw(user_input: dict) -> pd.DataFrame:
    """Uses a cascading lookup (Market -> Commodity) to fill all dependent features."""

    # 1. Start with the safest values (Global Defaults)
    full_data = OVERALL_DEFAULTS.copy()

    # 2. CASCADING: Location Features based on Market (Overrides defaults)
    market_name = str(user_input.get("market", "")).strip()
    if market_name in MARKET_LOOKUP_MAP:
        # Fills admin1, admin2, latitude, longitude
        full_data.update(MARKET_LOOKUP_MAP[market_name])

    # 3. CASCADING: Classification Features based on Commodity (Overrides defaults/market)
    commodity_name = str(user_input.get("commodity", "")).strip()
    if commodity_name in COMMODITY_LOOKUP_MAP:
        # Fills category, unit, pricetype
        full_data.update(COMMODITY_LOOKUP_MAP[commodity_name])

    # 4. USER INPUT: Overwrite all with the 5 provided features (Highest Priority)
    full_data.update(user_input)

    # Convert to appropriate types (using full_data for all values)
    lat = float(full_data.get("latitude", 0.0))
    lon = float(full_data.get("longitude", 0.0))
    year = int(full_data.get("year", 2025))
    month = int(full_data.get("month", 1))

    # 5. Start a blank feature row and fill numeric fields
    row = {col: 0.0 for col in TRAIN_COLUMNS}
    for name, val in [
        ("latitude", lat), ("longitude", lon),
        ("year", year), ("month", month),
    ]:
        if name in row:
            row[name] = float(val)

    # 6. Handle all categorical features with UNKNOWN FALLBACK
    categorical_keys = [
        "admin1", "admin2", "market", "category",
        "commodity", "unit", "pricetype"
    ]

    for key in categorical_keys:
        value = str(full_data.get(key, "")).strip()

        is_known = value in KNOWN_CATEGORIES.get(key, [])

        if is_known:
            col_name = f"{key}_{value}"
        else:
            col_name = f"{key}_OTHER"  # Adaptive fallback

        if col_name in row:
            row[col_name] = 1.0

    # 7. Recreate the engineered features
    row["price_per_lat"] = 1.0 / (abs(lat) + 1e-6)
    row["price_per_long"] = 1.0 / (abs(lon) + 1e-6)

    # 8. Convert to DataFrame and scale
    df_one = pd.DataFrame([row], columns=TRAIN_COLUMNS)

    if hasattr(SCALER, "feature_names_in_"):
        cols_to_scale = list(SCALER.feature_names_in_)
        cols_to_scale_present = [
            c for c in cols_to_scale if c in df_one.columns]

        if cols_to_scale_present:
            df_one[cols_to_scale_present] = SCALER.transform(
                df_one[cols_to_scale_present])

    return df_one
