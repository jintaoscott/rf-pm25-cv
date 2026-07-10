# -*- coding: utf-8 -*-
"""Rebuild the daily PM2.5 input table from the public CNEMC archive.

The model pipeline consumes a daily station-level PM2.5 table
(columns: index, Code, Value, datetime). The raw hourly observations are
published by the China National Environmental Monitoring Centre (CNEMC) and
archived as daily files at https://quotsoft.net/air/ (Wang Xiaolei):

    https://quotsoft.net/air/data/china_sites_YYYYMMDD.csv

Each file is a wide CSV (date, hour, type, <station codes...>) with one row
per metric type and hour. This script downloads the files for a date range,
keeps rows with type == "PM2.5" (hourly concentrations; NOT the PM2.5_24h
moving average), and aggregates per station-day:

    Value = arithmetic mean over the available (finite, non-blank) hours;
    a station-day is kept if at least one hour is available.

This rule reproduces the original study input (verification details in
DATA.md). Use --verify to check a rebuilt table against the published
per-station annual statistics.

Behaviour on failure: downloads are retried, written atomically, and
validated (header, row shape, row dates, PM2.5 rows, numeric cells). If any
requested date cannot be fetched or fails validation, the final table is
NOT written and the script exits non-zero (override with --allow-missing).
With --verify, verification runs before the final table is written; the
output itself is written atomically.

Usage:
    python tools/fetch_rebuild_pm25.py 2015-01-01 2019-12-31 \
        -o data/2015_2019AQData.csv --verify

The inclusive 2015-2019 range contains 1,826 requested dates (~3 GB);
please keep the default politeness delay at that scale.
"""
import argparse
import csv
import math
import os
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import date, timedelta

URL = "https://quotsoft.net/air/data/china_sites_{ymd}.csv"
UA = {"User-Agent": "Mozilla/5.0 (research reproduction script)"}
RETRIES = 3

# 157 national-network stations used in the study: (pipeline index, CNEMC code)
ROSTER = [
    (0, "1001A"), (1, "1002A"), (2, "1003A"), (3, "1004A"),
    (4, "1005A"), (5, "1006A"), (6, "1007A"), (7, "1008A"),
    (8, "1009A"), (9, "1010A"), (10, "1011A"), (11, "1012A"),
    (12, "1013A"), (13, "1014A"), (14, "1015A"), (15, "1016A"),
    (16, "1017A"), (17, "1018A"), (18, "1019A"), (19, "1020A"),
    (20, "1021A"), (21, "1023A"), (22, "1024A"), (23, "1025A"),
    (24, "1026A"), (25, "1027A"), (26, "1036A"), (27, "1037A"),
    (28, "1038A"), (29, "1039A"), (30, "1040A"), (31, "1041A"),
    (32, "1042A"), (33, "1043A"), (34, "1044A"), (35, "1045A"),
    (36, "1046A"), (37, "1047A"), (38, "1048A"), (39, "1049A"),
    (40, "1050A"), (41, "1051A"), (42, "1052A"), (43, "1053A"),
    (44, "1054A"), (45, "1055A"), (46, "1057A"), (47, "1058A"),
    (48, "1059A"), (49, "1060A"), (50, "1061A"), (51, "1062A"),
    (52, "1063A"), (53, "1064A"), (54, "1065A"), (55, "1066A"),
    (56, "1067A"), (57, "1069A"), (58, "1070A"), (59, "1071A"),
    (60, "1072A"), (61, "1073A"), (62, "1074A"), (63, "1075A"),
    (64, "1077A"), (65, "1078A"), (66, "1079A"), (67, "1081A"),
    (68, "1082A"), (69, "1083A"), (70, "1084A"), (71, "1085A"),
    (72, "1086A"), (73, "1087A"), (74, "1088A"), (75, "1089A"),
    (76, "1299A"), (77, "1300A"), (78, "1302A"), (79, "1303A"),
    (80, "1304A"), (81, "1306A"), (82, "1316A"), (83, "1317A"),
    (84, "1318A"), (85, "1319A"), (86, "1320A"), (87, "1321A"),
    (88, "1323A"), (89, "1324A"), (90, "1622A"), (91, "1623A"),
    (92, "1624A"), (93, "1625A"), (94, "1626A"), (95, "1627A"),
    (96, "1629A"), (97, "1630A"), (98, "1632A"), (99, "1633A"),
    (100, "1634A"), (101, "1635A"), (102, "1636A"), (103, "1653A"),
    (104, "1655A"), (105, "1718A"), (106, "1719A"), (107, "1720A"),
    (108, "1727A"), (109, "1729A"), (110, "1730A"), (111, "1731A"),
    (112, "1738A"), (113, "1739A"), (114, "1740A"), (115, "1741A"),
    (116, "1742A"), (117, "1743A"), (118, "1818A"), (119, "1819A"),
    (120, "1820A"), (121, "1822A"), (122, "1823A"), (123, "1824A"),
    (124, "1825A"), (125, "1826A"), (126, "1827A"), (127, "1828A"),
    (128, "1829A"), (129, "1830A"), (130, "2160A"), (131, "2161A"),
    (132, "2163A"), (133, "2164A"), (134, "2165A"), (135, "2386A"),
    (136, "2387A"), (137, "2388A"), (138, "2389A"), (139, "2391A"),
    (140, "2393A"), (141, "2394A"), (142, "2395A"), (143, "2845A"),
    (144, "2858A"), (145, "2859A"), (146, "2860A"), (147, "2878A"),
    (148, "2919A"), (149, "2922A"), (150, "3020A"), (151, "3021A"),
    (152, "3051A"), (153, "3054A"), (154, "3066A"), (155, "3132A"),
    (156, "3141A"),
]
CODES = {code for _, code in ROSTER}
INDEX_OF = {code: i for i, code in ROSTER}


class DayFileError(Exception):
    pass


def validate_day_file(path, ymd):
    """Cheap structural checks; raises DayFileError on problems."""
    with open(path, "r", encoding="utf-8", errors="strict", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise DayFileError("empty file")
        if header[:3] != ["date", "hour", "type"]:
            raise DayFileError("unexpected header %r" % header[:3])
        roster_cols = [c for c in header[3:] if c in CODES]
        if not roster_cols:
            raise DayFileError("no study-roster station columns")
        if len(roster_cols) != len(set(roster_cols)):
            raise DayFileError("duplicate station columns")
        saw_pm25 = False
        for row in reader:
            if len(row) != len(header):
                raise DayFileError("row length %d != header %d" % (len(row), len(header)))
            if row[0] != ymd:
                raise DayFileError("row date %r != requested %s" % (row[0], ymd))
            if row[2] == "PM2.5":
                saw_pm25 = True
        if not saw_pm25:
            raise DayFileError("no PM2.5 rows")


def fetch_day(d, cache_dir, delay):
    """Return path to a validated cached file, or None if unobtainable."""
    ymd = d.strftime("%Y%m%d")
    path = os.path.join(cache_dir, "china_sites_%s.csv" % ymd)
    if os.path.exists(path):
        try:
            validate_day_file(path, ymd)
            return path
        except DayFileError as e:
            print("  cache invalid for %s (%s), re-downloading" % (ymd, e))
            os.remove(path)
    last_err = None
    for attempt in range(1, RETRIES + 1):
        req = urllib.request.Request(URL.format(ymd=ymd), headers=UA)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            part = path + ".part"
            with open(part, "wb") as f:
                f.write(data)
            validate_day_file(part, ymd)
            os.replace(part, path)
            return path
        except Exception as e:
            last_err = e
            time.sleep(delay * attempt)
    print("  MISS %s after %d attempts (%s)" % (ymd, RETRIES, last_err))
    return None


def daily_means(path, ymd):
    """station code -> [sum, n_hours] over type == PM2.5 rows."""
    acc = defaultdict(lambda: [0.0, 0])
    seen_hours = set()
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        cols = [(j, c) for j, c in enumerate(header) if c in CODES]
        for row in reader:
            if len(row) > 2 and row[2] == "PM2.5":
                hour = row[1]
                if hour in seen_hours:
                    raise DayFileError("duplicate PM2.5 hour %s on %s" % (hour, ymd))
                seen_hours.add(hour)
                for j, code in cols:
                    if j < len(row) and row[j] != "":
                        try:
                            v = float(row[j])
                        except ValueError:
                            raise DayFileError(
                                "non-numeric cell %r for %s on %s" % (row[j], code, ymd))
                        if not math.isfinite(v):
                            raise DayFileError(
                                "non-finite value %r for %s on %s" % (row[j], code, ymd))
                        acc[code][0] += v
                        acc[code][1] += 1
    return acc


def verify_against_annual_stats(rows, d0, d1):
    """Compare rebuilt per-station annual stats with the published reference
    for every complete calendar year in [d0, d1]. Returns True if all pass."""
    ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            os.pardir, "results", "pm25_station_annual_stats.csv")
    if not os.path.exists(ref_path):
        print("verify: reference file not found (%s), skipping" % ref_path)
        return True
    full_years = [y for y in range(d0.year, d1.year + 1)
                  if d0 <= date(y, 1, 1) and date(y, 12, 31) <= d1]
    if not full_years:
        print("verify: no complete calendar year in range, nothing to check")
        return True
    per = defaultdict(list)  # (code, year) -> values
    for idx, code, value, iso in rows:
        y = int(iso[:4])
        if y in full_years:
            per[(code, y)].append(value)
    ref = {}
    with open(ref_path, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            ref[(r["Code"], int(r["year"]))] = r
    ok = True
    for y in full_years:
        n_checked = n_bad = 0
        keys = [k for k in ref if k[1] == y]
        for k in keys:
            vals = per.get(k)
            r = ref[k]
            n_checked += 1
            if vals is None:
                n_bad += 1
                continue
            n = len(vals)
            mean = sum(vals) / n
            var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
            checks = (n == int(r["n_days"]),
                      abs(mean - float(r["mean"])) <= 0.011,
                      abs(var ** 0.5 - float(r["std"])) <= 0.011,
                      abs(min(vals) - float(r["min"])) <= 0.011,
                      abs(max(vals) - float(r["max"])) <= 0.011)
            if not all(checks):
                n_bad += 1
        extra = {k for k in per if k[1] == y} - set(keys)
        status = "PASS" if (n_bad == 0 and not extra) else "FAIL"
        if status == "FAIL":
            ok = False
        print("verify %d: %s (%d station-years checked, %d mismatched, %d unexpected)"
              % (y, status, n_checked, n_bad, len(extra)))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("start", help="YYYY-MM-DD")
    ap.add_argument("end", help="YYYY-MM-DD")
    ap.add_argument("-o", "--out", default="pm25_daily_rebuilt.csv")
    ap.add_argument("--cache", default="china_sites_cache")
    ap.add_argument("--delay", type=float, default=0.5)
    ap.add_argument("--allow-missing", action="store_true",
                    help="write the output even if some dates could not be fetched")
    ap.add_argument("--verify", action="store_true",
                    help="check complete years against results/pm25_station_annual_stats.csv")
    args = ap.parse_args()

    d0 = date.fromisoformat(args.start)
    d1 = date.fromisoformat(args.end)
    if d1 < d0:
        ap.error("end date is before start date")
    os.makedirs(args.cache, exist_ok=True)

    rows, missing = [], []
    d = d0
    while d <= d1:
        ymd = d.strftime("%Y%m%d")
        path = fetch_day(d, args.cache, args.delay)
        if path is None:
            missing.append(ymd)
        else:
            try:
                for code, (s, n) in sorted(daily_means(path, ymd).items()):
                    if n > 0:
                        rows.append((INDEX_OF[code], code, s / n, d.isoformat()))
            except DayFileError as e:
                print("  INVALID %s (%s)" % (ymd, e))
                missing.append(ymd)
        d += timedelta(days=1)
        time.sleep(args.delay)

    n_req = (d1 - d0).days + 1
    print("requested %d dates | ok %d | missing %d"
          % (n_req, n_req - len(missing), len(missing)))
    if missing:
        print("missing dates:", " ".join(missing))
        if not args.allow_missing:
            print("output NOT written (use --allow-missing to override)")
            sys.exit(2)

    if args.verify and not verify_against_annual_stats(rows, d0, d1):
        print("verification FAILED - output not written")
        sys.exit(3)

    part = args.out + ".part"
    with open(part, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "Code", "Value", "datetime"])
        w.writerows(rows)
    os.replace(part, args.out)
    print("OK: %d station-days -> %s" % (len(rows), args.out))


if __name__ == "__main__":
    main()
