"""
Centralized configuration for RF PM2.5 cross-validation study.
All feature lists, model variants, best parameters, and paths live here.
"""
from pathlib import Path
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(exist_ok=True)

# ── Raw data files ─────────────────────────────────────────────────────
FILES = {
    "aq":         DATA_DIR / "2015_2019AQData.csv",
    "weather":    DATA_DIR / "Weather2015_2019_Stations_daily.csv",
    "aod1":       DATA_DIR / "aod_sites.csv",
    "aod2":       DATA_DIR / "aod_sites_2.csv",
    "landuse":    DATA_DIR / "landuse.csv",
    "road":       DATA_DIR / "road.csv",
    "population": DATA_DIR / "population.csv",
}

# ── Weather columns from ERA5 ─────────────────────────────────────────
WEATHER_COLS = ["BLH", "SP", "TCC", "VAR_10U", "VAR_10V", "VAR_2D", "VAR_2T"]

# ── 15 selected land-use / road features (after VIF filtering) ────────
LANDUSE_ROAD_COLS = [
    "path_road_2000m", "unclassified_road_100m", "residential_road_2000m",
    "track_road_2000m", "path_road_1000m", "steps_road_2000m",
    "forest_2000m", "secondary_link_road_1000m", "pedestrian_road_1500m",
    "commercial_1000m", "tertiary_link_road_1500m", "tertiary_road_2000m",
    "footway_road_1500m", "motorway_road_1500m", "trunk_link_road_1000m",
]

# ── Feature sets for each model variant (Table 1 in paper) ────────────
FEATURES = {
    "non_aod": WEATHER_COLS + LANDUSE_ROAD_COLS,            # 22 features
    "aod":     WEATHER_COLS + LANDUSE_ROAD_COLS + ["AOD", "AOD_a"],  # 24 features
    "date_id": ["dateid", "latitude", "longitude"],          # 3 features
}
# distance-controlled uses same features as non_aod, just fewer stations

# ── 40-station subset for Distance-controlled model ───────────────────
# Selected via K-means clustering of 157 stations into 40 clusters,
# picking the station closest to each cluster centroid.
DISTANCE_CONTROLLED_STATIONS = [
    "1317A", "1075A", "1016A", "1081A", "1820A", "1629A", "1059A", "1063A",
    "1655A", "1006A", "1045A", "1626A", "1051A", "1038A", "2160A", "1742A",
    "1302A", "1720A", "1047A", "1073A", "3021A", "3132A", "1729A", "1623A",
    "1823A", "1632A", "1828A", "2919A", "1026A", "2389A", "1008A", "2387A",
    "1079A", "1633A", "1010A", "1024A", "1025A", "1306A", "1089A", "1635A",
]

# ── CV settings ────────────────────────────────────────────────────────
CV_BLOCK_COLUMNS = {
    "spatial":  "index",          # block by station
    "temporal": "random_dateid",  # block by randomized date
}

# ── Optuna winners (step04, Non-AOD; must match results/optuna_summary.txt) ──
OPTUNA_WINNERS = {
    "spatial": {
        "max_features": 20, "max_samples": 0.998370137990351,
        "max_depth": 32, "min_samples_split": 3,
        "n_estimators": 90, "random_state": 42, "n_jobs": -1,
    },
    "temporal": {
        "max_features": 3, "max_samples": 0.9705474591587601,
        "max_depth": 40, "min_samples_split": 2,
        "n_estimators": 90, "random_state": 42, "n_jobs": -1,
    },
}

# ── Default RF parameters ──────────────────────────────────────────────
DEFAULT_RF_PARAMS = {
    "random_state": 42,
    "n_jobs": -1,
}

# ── Grids for the multi-year single-parameter sweeps (step08) ─────────────
SWEEP_RANGES = {
    "max_features":      {"range": (1, 23),  "step": 1,    "dtype": int},
    "min_samples_split": {"range": (2, 35),  "step": 1,    "dtype": int},
    "max_depth":         {"range": (5, 36),  "step": 1,    "dtype": int},
    "max_samples":       {"range": (0.02, 1.01), "step": 0.02, "dtype": float},
}

# ── Target variable ───────────────────────────────────────────────────
TARGET = "Value"
