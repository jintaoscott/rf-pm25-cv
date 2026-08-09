# Data sources and assembly

Six of the seven model-input tables are included in `data/`, redistributed
under their providers' open licences (see attributions below). The PM2.5
observations are NOT redistributed; obtain them from CNEMC as described
below and place the file in `data/`.

| dataset | provider | notes |
|---|---|---|
| Hourly PM2.5 (2015-2019) | China National Environmental Monitoring Centre (CNEMC), http://www.cnemc.cn | 157 national-network stations in the study region in northern China (station codes like `1001A`); we aggregate to daily means |
| Satellite AOD | NASA MODIS Terra/Aqua, SDS AOD_550_Dark_Target_Deep_Blue_Combined, via NASA Earth Observations (NEO) | 0.1 deg; one feature per satellite |
| Meteorology | ERA5 reanalysis (ECMWF Copernicus Climate Change Service) | hourly, 0.25 deg; 7 variables (BLH, SP, TCC, 10U, 10V, 2D, 2T) |
| Land use / roads | OpenStreetMap | buffer statistics (land use: 100, 500, 1000, 1500, 2000 m; roads: 50, 100, 500, 1000, 1500, 2000 m) |
| Population density | WorldPop | 1 km, yearly |

Included in `data/`, all covering the 157 national stations:
`Weather2015_2019_Stations_daily.csv` (ERA5-derived), `aod_sites.csv` and
`aod_sites_2.csv` (MODIS-derived), `landuse.csv` and `road.csv`
(OpenStreetMap-derived), `population.csv` (WorldPop-derived).
To be obtained by the user: `2015_2019AQData.csv`, the daily station-level
PM2.5 table (daily means derived from CNEMC hourly observations; see
"Rebuilding the PM2.5 table" below). Step 00 checks the merged per-year
sample, station, and feature counts before any model fitting.

## Rebuilding the PM2.5 table

CNEMC does not offer a bulk historical download; its published observations
are archived by third parties. `tools/fetch_rebuild_pm25.py` rebuilds the
daily table from Wang Xiaolei's historical archive of the CNEMC real-time
feed (https://quotsoft.net/air/, maintained since May 2014):

    python tools/fetch_rebuild_pm25.py 2015-01-01 2019-12-31 -o data/2015_2019AQData.csv

Aggregation rule: daily arithmetic mean over the available hourly PM2.5
values; a station-day is kept if at least one hour is available. The
authors verified the rebuild against the original study input for January
2019: all 4,353 rebuilt station-days matched within 1e-6, and the 514
station-days with no reported hours corresponded exactly to the empty
values in the original. Readers can check a full rebuild against
`results/pm25_station_annual_stats.csv` (the script's `--verify` flag does
this automatically). The inclusive range contains 1,826 requested dates
(~3 GB). The rebuilt table (columns index, Code, Value,
datetime) is a drop-in replacement: the pipeline reads only these four
columns.

An independent, redistributable alternative is Bai et al. (2020), 'A
homogenized daily in situ PM2.5 concentration dataset in China during
2015-2019', PANGAEA, https://doi.org/10.1594/PANGAEA.917557 (CC BY 4.0;
1,309 stations). It was scraped from the same CNEMC feed independently of
the Wang archive; in the authors' comparison, coordinate-matched stations
agree with the table rebuilt from the Wang archive at r = 0.99 overall
with mean bias of about -0.3 ug/m3. It is bias-adjusted, integer-rounded,
and identifies stations by coordinates rather than CNEMC codes, so it
supports independent statistical replication rather than exact
reproduction of this study's input.

## How the feature tables were derived

- `Weather2015_2019_Stations_daily.csv`: hourly ERA5 surface fields (0.25 deg;
  BLH, SP, TCC, 10U, 10V, 2D, 2T) covering 34.75-41.0 N, 112.5-119.75 E were
  extracted at the station locations, converted from UTC to local time
  (Asia/Shanghai), and averaged to daily means per station.
- `aod_sites.csv` / `aod_sites_2.csv`: daily MODIS Terra / Aqua AOD grids
  (0.1 deg) exported from NASA Earth Observations (NEO) as GeoTIFF were
  sampled at the station locations. The NEO no-data value (255) is encoded
  as the numeric sentinel 99999, which the AOD model retains as an input
  value (the operational treatment described in the paper). One table per
  satellite.
- `landuse.csv` / `road.csv`: land-use polygon areas and road lengths from
  the Geofabrik "china-latest" shapefile extract (gis_osm_landuse_a_free_1,
  gis_osm_roads_free_1; extract retrieved 2023-04-18, i.e. OSM state of
  mid-April 2023) intersected with buffers around each station: 100-2000 m
  for land use, 50-2000 m for roads. Treated as static features.
- `population.csv`: WorldPop China population density (1 km, yearly,
  chn_pd_YYYY_1km.tif) sampled at the station coordinates, one value per
  year 2015-2019.

## Integrity of the distributed tables

The six auxiliary tables are row/column subsets of the study's original
feature tables, produced by text-level subsetting: rows are restricted to
the 157 national-network stations and unnecessary identifying and metadata
columns are dropped, but
every retained field keeps its exact original byte string. The
`latitude`/`longitude` columns of `landuse.csv` and `road.csv` are part of
the model input, not metadata: through the merge order in
`src/data.py::load_dataset` they are the station coordinates the Date-ID
model uses. Design matrices built from these auxiliary tables are identical to
those built from the full originals. With the original PM2.5 table and row
order, the result tables reproduce at the reported precision; see README,
"Reproducing the experiments". SHA-256 of the distributed
files:

```
810bd9d1256b8d59b57c785f3d08ac3afa069a8e6fef93e818892ff103b309d4  Weather2015_2019_Stations_daily.csv
dbaf6054c76a666ae8b759163f3c80d238fadf75273e66083295d2d178988244  aod_sites.csv
bda7423f35eb26dfcb25a7545583b59882ac85a4b171b5d0384b214ba3b03b6c  aod_sites_2.csv
923a2e670d9bd09258b1000fa5352f3d89223319a29a30dd7395fb58ee8f2eef  landuse.csv
5f108dc7ba5f72ddf4ce6fd0ac5faf2e88174cf9c23217a76b90aeec50316401  road.csv
aadd0e0f3af11977019a77776a24f2eb15d3e49af95eb89261785547e04f584d  population.csv
```

## Attributions for redistributed derived data

- Weather variables: generated using Copernicus Climate Change Service
  information (ERA5); neither the European Commission nor ECMWF is
  responsible for any use of this information.
- AOD: derived from NASA MODIS products via NASA Earth Observations; NASA
  data are not subject to copyright restrictions.
- Land-use and road statistics: (c) OpenStreetMap contributors, derived
  database redistributed under the Open Database License (ODbL).
- Population density: derived from WorldPop (www.worldpop.org), CC BY 4.0.

`results/station_counts_by_year.csv` documents the effective station counts
per year after quality filtering (139-148 of the 157-station roster).

`results/pm25_station_annual_stats.csv` gives per-station annual summary
statistics (n_days, mean, std, min, max) of the daily PM2.5 input, as
aggregate descriptors and as a verification target: after rebuilding the
PM2.5 table with `tools/fetch_rebuild_pm25.py`, your per-station annual
statistics should match this file.
