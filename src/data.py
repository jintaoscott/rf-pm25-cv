"""
Data loading and preparation.
Merges AQ, weather, AOD, land-use, road, and population CSVs.
"""
import pandas as pd
import numpy as np
from . import config


def _load_weather():
    df = pd.read_csv(config.FILES["weather"])
    return df[["date", "station_idx", "latitude", "longitude"] + config.WEATHER_COLS]


def load_dataset(year=None, include_aod=True):
    """
    Load and merge all data sources.

    Parameters
    ----------
    year : int or None
        If given, filter to that year only.
    include_aod : bool
        Whether to merge AOD columns.

    Returns
    -------
    pd.DataFrame with columns:
        index, Code, station, Value, datetime, date, year,
        latitude, longitude, weather vars, land-use/road vars,
        population_density, dateid, daily_avg, random_dateid,
        (and AOD, AOD_a if include_aod=True)
    """
    # AQ + weather
    df = pd.read_csv(config.FILES["aq"])
    # Analysis scope: national monitoring network only (station codes like
    # '1001A'); rows without a valid national station code are dropped.
    df = df[df["Code"].notna() & (df["Code"].astype(str) != "0")]
    df_weather = _load_weather()
    df = df.merge(df_weather, left_on=["datetime", "index"],
                  right_on=["date", "station_idx"], how="left")

    # AOD
    if include_aod:
        for key in ("aod1", "aod2"):
            dfaod = pd.read_csv(config.FILES[key])
            df = df.merge(dfaod, on=["date", "index"], how="left")

    # Land-use + road (static, merge on station index only)
    dfroad = pd.read_csv(config.FILES["road"])
    dfland = pd.read_csv(config.FILES["landuse"])
    # Drop spurious index columns from CSVs
    for _df in (dfroad, dfland):
        for col in list(_df.columns):
            if col.startswith("Unnamed"):
                _df.drop(columns=col, inplace=True)
    df = df.merge(dfroad, on="index", how="left")
    df = df.merge(dfland, on="index", how="left")

    # Population density (yearly)
    dfp = pd.read_csv(config.FILES["population"])
    dfp = dfp.melt(id_vars=["index"], var_name="year_str", value_name="population_density")
    dfp["year"] = dfp["year_str"].str.extract(r"(\d+)").astype(int)
    dfp = dfp.drop(columns="year_str")

    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df = df.merge(dfp, on=["index", "year"], how="left")

    # Year filter
    if year is not None:
        df = df[df["year"] == year]

    # Derived columns
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")
    df["dateid"] = df["datetime"].factorize()[0]

    daily_avg = df.groupby("datetime")[config.TARGET].mean()
    df = df.set_index("datetime")
    df["daily_avg"] = daily_avg
    df = df.reset_index()

    unique_dates = df["datetime"].unique()
    unique_dates = np.array(unique_dates)  # ensure proper numpy array for shuffle
    rng = np.random.RandomState(42)
    rng.shuffle(unique_dates)
    df["random_dateid"] = df["datetime"].map(
        {d: i for i, d in enumerate(unique_dates)}
    )

    return df


def prepare_features(df, model_variant="non_aod", station_subset=None):
    """
    Select features, filter stations, drop NAs, return clean dfx.

    Parameters
    ----------
    df : pd.DataFrame from load_dataset()
    model_variant : str
        One of "non_aod", "aod", "date_id"
    station_subset : list or None
        If given, filter df to these station Codes (for distance-controlled model).

    Returns
    -------
    dfx : pd.DataFrame ready for BlockCV
    features : list of feature column names
    """
    if station_subset is not None:
        df = df[df["Code"].isin(station_subset)]

    features = config.FEATURES[model_variant]
    keep_cols = features + [config.TARGET, "index", "random_dateid",
                            "Code", "latitude", "longitude", "dateid"]
    # Only keep columns that exist in df, and only once each: the date_id feature
    # list overlaps the bookkeeping columns, and selecting a repeated name from a
    # DataFrame returns every matching column.
    keep_cols = list(dict.fromkeys(c for c in keep_cols if c in df.columns))
    dfx = df[keep_cols].dropna()

    return dfx, features
