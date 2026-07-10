"""
STEP 00 - Verify the national-only filter before running anything else.

Confirms the national-network scope filter in src/data.py and that
per-year sample counts match previously established fingerprints for
this dataset. Every later step assumes
these numbers; if any assert fails here, STOP and investigate.

Runtime: ~2-4 minutes (data loading only, no model fits).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

from src.data import load_dataset, prepare_features

# (N_rows, N_stations) for the non_aod feature set after dropna, per year,
# national stations only - measured with the filtered pipeline locally.
EXPECTED = {
    2015: (50451, 143),
    2016: (49390, 139),
    2017: (51635, 148),
    2018: (50519, 143),
    2019: (50551, 141),
}

print("Loading full dataset (all years, no AOD)...", flush=True)
df = load_dataset(year=None, include_aod=False)

n_stations_raw = df["index"].nunique()
assert df["index"].max() < 157, (
    f"index values >=157 present (max={df['index'].max()}) - filter NOT active"
)
assert not (df["Code"].astype(str) == "0").any(), "non-national station rows present - filter NOT active"
print(f"raw merged rows: {len(df):,} | distinct stations: {n_stations_raw} (must be <=157)")

ok = True
for year in sorted(EXPECTED):
    dfy = df[df["year"] == year]
    dfx, feats = prepare_features(dfy, "non_aod")
    n, s = len(dfx), dfx["index"].nunique()
    en, es = EXPECTED[year]
    mark = "OK " if (n, s) == (en, es) else "FAIL"
    if (n, s) != (en, es):
        ok = False
    print(f"{year}: N={n:,} stations={s}   expected N={en:,}/{es}   [{mark}]", flush=True)
    assert len(feats) == 22

# AOD variant must have identical row counts (sentinel encoding keeps all rows)
df19 = load_dataset(year=2019, include_aod=True)
dfx19, feats19 = prepare_features(df19, "aod")
print(f"2019 aod-variant: N={len(dfx19):,} (expected 50,551) features={len(feats19)} (expected 24)")
assert len(dfx19) == 50551 and len(feats19) == 24

assert ok, "One or more yearly fingerprints failed"
print("\nALL FINGERPRINTS PASS - the pipeline is national-only. Proceed to step01.")
