"""
External parameters for flight price prediction (Step 1b of the report).

Adds four groups of features to the engineered flight table:

* calendar    - departure weekday, weekend flag, distance to the nearest public holiday
* weather     - climate normals for the departure day at origin and destination
                (10-year averages, i.e. information a traveller has in advance)
* oracle      - the weather actually observed on the departure day. This is NOT known
                when the fare is quoted, so it is only used as a labelled upper bound.
* competitor  - fares offered by *other* airlines on the same route, class and date

Every fare in Clean_Dataset.csv was quoted on a single search date, so
departure_date = search_date + days_left. Fuel prices and airline traffic
statistics in force on that one search date would be constant per origin city /
airline and therefore add no information beyond `source_city` / `airline`; they
are deliberately not included.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.utils import setup_logger

logger = setup_logger("external_features")

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY_VARS = ["precipitation_sum", "temperature_2m_max", "temperature_2m_min", "wind_speed_10m_max"]
HOURLY_VARS = ["relative_humidity_2m", "wind_speed_10m"]

CALENDAR_FEATURES = [
    "dep_weekday",
    "is_weekend_departure",
    "days_to_holiday",
    "abs_days_to_holiday",
    "is_holiday_window",
]
WEATHER_BASE = ["rain_mm", "tmax_c", "tmin_c", "wind_max_kmh", "fog_prob"]
CLIMATE_FEATURES = [f"{side}_clim_{v}" for side in ("src", "dst") for v in WEATHER_BASE]
ORACLE_FEATURES = [f"{side}_obs_{v}" for side in ("src", "dst") for v in WEATHER_BASE]
COMPETITOR_FEATURES = [
    "comp_min_log_fare",
    "comp_median_log_fare",
    "comp_n_airlines",
    "comp_n_flights",
    "own_airline_flights",
]


# --------------------------------------------------------------------------- dates
def add_dates(df: pd.DataFrame, search_date: str) -> pd.DataFrame:
    out = df.copy()
    base = pd.Timestamp(search_date)
    out["departure_date"] = base + pd.to_timedelta(out["days_left"].astype(int), unit="D")
    return out


# --------------------------------------------------------------------------- calendar
def add_calendar_features(df: pd.DataFrame, holidays: Dict[str, str], window: int) -> pd.DataFrame:
    out = df.copy()
    dep = out["departure_date"]
    out["dep_weekday"] = dep.dt.dayofweek.astype(int)  # Monday = 0
    out["is_weekend_departure"] = dep.dt.dayofweek.isin([4, 5, 6]).astype(int)  # Fri-Sun

    hol = np.array(sorted(pd.to_datetime(list(holidays.keys())).values))
    dep_vals = dep.values.astype("datetime64[D]")
    diffs = (hol.astype("datetime64[D]")[None, :] - dep_vals[:, None]).astype(int)  # holiday - departure
    nearest = diffs[np.arange(len(diffs)), np.abs(diffs).argmin(axis=1)]
    out["days_to_holiday"] = nearest  # >0: holiday ahead, <0: just after
    out["abs_days_to_holiday"] = np.abs(nearest)
    out["is_holiday_window"] = (np.abs(nearest) <= window).astype(int)
    return out


# --------------------------------------------------------------------------- weather
def _fetch_city(lat: float, lon: float, start: str, end: str) -> dict:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": ",".join(DAILY_VARS),
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "Asia/Kolkata",
    }
    url = f"{ARCHIVE_URL}?{urllib.parse.urlencode(params)}"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # network hiccup or rate limit
            logger.warning("Weather download failed (%s), retrying...", exc)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Could not download weather for ({lat}, {lon}).")


def _daily_table(payload: dict) -> pd.DataFrame:
    daily = pd.DataFrame(payload["daily"])
    daily["date"] = pd.to_datetime(daily.pop("time"))
    hourly = pd.DataFrame(payload["hourly"])
    hourly["time"] = pd.to_datetime(hourly["time"])
    # Radiation-fog proxy: early-morning (00-08h) humid and calm air.
    morning = hourly[hourly["time"].dt.hour <= 8].copy()
    morning["date"] = morning["time"].dt.normalize()
    agg = morning.groupby("date").agg(rh=("relative_humidity_2m", "mean"), wind=("wind_speed_10m", "mean"))
    agg["fog"] = ((agg["rh"] >= 90) & (agg["wind"] < 8)).astype(float)
    daily = daily.merge(agg[["fog"]], left_on="date", right_index=True, how="left")
    return daily.rename(
        columns={
            "precipitation_sum": "rain_mm",
            "temperature_2m_max": "tmax_c",
            "temperature_2m_min": "tmin_c",
            "wind_speed_10m_max": "wind_max_kmh",
            "fog": "fog_prob",
        }
    )


def load_weather(
    cities: Dict[str, List[float]],
    cache_dir: Path,
    clim_years: List[int],
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> pd.DataFrame:
    """Daily weather per city from clim_years[0] to the end of the departure window (cached as CSV)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    start = f"{clim_years[0]}-01-01"
    end = window_end.strftime("%Y-%m-%d")
    for city, (lat, lon) in cities.items():
        path = cache_dir / f"weather_{city.lower()}.csv"
        if path.exists():
            table = pd.read_csv(path, parse_dates=["date"])
        else:
            logger.info("Downloading weather for %s (%s to %s)...", city, start, end)
            table = _daily_table(_fetch_city(lat, lon, start, end))
            table.to_csv(path, index=False)
        table["city"] = city
        frames.append(table)
    return pd.concat(frames, ignore_index=True)


def build_weather_lookup(
    weather: pd.DataFrame,
    clim_years: List[int],
    smoothing: int,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> pd.DataFrame:
    """Per (city, date in window): climate normal (mean over clim_years, +-smoothing days) and observation."""
    rows = []
    dates = pd.date_range(window_start, window_end, freq="D")
    hist = weather[weather["date"].dt.year.between(clim_years[0], clim_years[1])].copy()
    hist["doy"] = hist["date"].dt.dayofyear
    for city, city_hist in hist.groupby("city"):
        obs = weather[(weather["city"] == city)].set_index("date")
        for d in dates:
            doy = d.dayofyear
            near = city_hist[(city_hist["doy"] - doy).abs() <= smoothing]
            rec = {"city": city, "date": d}
            for v in WEATHER_BASE:
                rec[f"clim_{v}"] = float(near[v].mean())
                rec[f"obs_{v}"] = float(obs.loc[d, v]) if d in obs.index else np.nan
            rows.append(rec)
    return pd.DataFrame(rows)


def add_weather_features(df: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for side, col in (("src", "source_city"), ("dst", "destination_city")):
        renamed = lookup.rename(
            columns={c: f"{side}_{c}" for c in lookup.columns if c.startswith(("clim_", "obs_"))}
        )
        out = out.merge(
            renamed, how="left", left_on=[col, "departure_date"], right_on=["city", "date"]
        ).drop(columns=["city", "date"])
    return out


# --------------------------------------------------------------------------- competitors
def add_competitor_features(df: pd.DataFrame, target_col: str = "price") -> pd.DataFrame:
    """
    For each fare, summarise what *other* airlines charge on the same route, class and
    departure date (all quoted on the same search date, i.e. visible on screen together).
    The row's own airline is excluded, so no row sees its own price or its airline's prices.
    """
    out = df.copy()
    key = ["route", "class", "days_left"]
    log_price = np.log1p(out[target_col].astype(float))
    out["_lp"] = log_price
    comp_min = np.full(len(out), np.nan)
    comp_med = np.full(len(out), np.nan)
    comp_air = np.zeros(len(out))
    comp_n = np.zeros(len(out))
    own_n = np.zeros(len(out))

    for _, grp in out.groupby(key, sort=False):
        airlines = grp["airline"].to_numpy()
        prices = grp["_lp"].to_numpy()
        idx = grp.index.to_numpy()
        for airline in np.unique(airlines):
            mine = airlines == airline
            others = prices[~mine]
            pos = idx[mine]
            own_n[pos] = mine.sum()
            if others.size:
                comp_min[pos] = others.min()
                comp_med[pos] = np.median(others)
                comp_air[pos] = len(np.unique(airlines[~mine]))
                comp_n[pos] = others.size

    out["comp_min_log_fare"] = comp_min
    out["comp_median_log_fare"] = comp_med
    out["comp_n_airlines"] = comp_air
    out["comp_n_flights"] = comp_n
    out["own_airline_flights"] = own_n
    return out.drop(columns=["_lp"])


# --------------------------------------------------------------------------- entry point
def add_external_features(df: pd.DataFrame, raw_config: dict, project_root: Path) -> pd.DataFrame:
    """Attach all external parameter groups described in config.yaml -> external."""
    ext = raw_config["external"]
    out = add_dates(df.reset_index(drop=True), ext["search_date"])
    start, end = out["departure_date"].min(), out["departure_date"].max()
    logger.info("Departure window: %s to %s", start.date(), end.date())

    out = add_calendar_features(out, ext["holidays"], int(ext["holiday_window_days"]))

    years = list(ext["climatology_years"])
    weather = load_weather(ext["city_coordinates"], project_root / ext["weather_cache_dir"], years, start, end)
    lookup = build_weather_lookup(weather, years, int(ext["climatology_smoothing_days"]), start, end)
    out = add_weather_features(out, lookup)

    out = add_competitor_features(out)
    n_missing = int(out[CLIMATE_FEATURES + CALENDAR_FEATURES].isna().sum().sum())
    logger.info("External features added (%d missing calendar/climate values).", n_missing)
    return out
