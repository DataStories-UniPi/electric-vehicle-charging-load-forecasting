#!/usr/bin/env python
# coding: utf-8

# =========================================================
# LIBRARIES
# =========================================================
import os
import pickle
import warnings
from typing import Iterator, Dict, Any

import numpy as np
import pandas as pd
import holidays
import pytz


# =========================================================
# LOAD DATA
# =========================================================
# Only the columns we actually need
columns_to_load = ['Start Date', 'Start Time', 'End Date', 'End Time', 'CP ID', 'Total kWh', 'Site']

df1 = pd.read_csv('../../data/data_raw/raw_perth/September 2016 - August 2017.csv',
                  usecols=columns_to_load)
df2 = pd.read_csv('../../data/data_raw/raw_perth/September 2017-August 2018.csv',
                  usecols=columns_to_load)
df3 = pd.read_csv('../../data/data_raw/raw_perth/September 2018-August 2019.csv',
                  usecols=columns_to_load)


# =========================================================
# PREPROCESS DATA
# =========================================================
def process_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine 'Start Date' + 'Start Time' and 'End Date' + 'End Time'
    into timestamp columns, then sort by start.
    """
    df = df.copy()
    df["Start Date"] = pd.to_datetime(df["Start Date"].str[:10] + " " + df["Start Time"])
    df["End Date"]   = pd.to_datetime(df["End Date"].str[:10]   + " " + df["End Time"])
    return df.sort_values(by="Start Date")

# Apply to all three files and concatenate
df1 = process_df(df1)
df2 = process_df(df2)
df3 = process_df(df3)
df  = pd.concat([df1, df2, df3], ignore_index=True)

# Drop raw time columns; we keep the combined datetimes
df = df.drop(columns=['Start Time', 'End Time'])

# QA counters (not printed)
missing_values = df.isnull().sum()
negative_energy_count = (df['Total kWh'] < 0).sum()

# Keep non-negative energy, valid end timestamps, and durations > 0
df = df[df['Total kWh'] >= 0]
df = df.dropna(subset=['End Date'])

# Re-check missing values (for QA)
missing_values = df.isnull().sum()

# Positive duration only
df = df[df['End Date'] > df['Start Date']]


# =========================================================
# TIME SERIES PROCESSING (expand sessions → 10-min intervals)
# =========================================================
# City label
df['City'] = 'Perth'

def expand_sessions_to_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand each charging session into 10-minute intervals, evenly distributing
    the session's total energy across its intervals.

    Returns a DataFrame with one row per 10-minute slice:
      ['Timestamp', 'Total kWh', 'CP ID', 'City', 'Site']
    """
    # Duration of each session in 10-minute steps (integer)
    durations = ((df['End Date'] - df['Start Date']).dt.total_seconds() / 600).astype(int)

    # Pre-allocate arrays for speed (same semantics as original)
    total_rows = durations.sum()
    timestamps    = np.empty(total_rows, dtype='datetime64[m]')
    energy_values = np.empty(total_rows, dtype=float)
    station_names = np.empty(total_rows, dtype=object)
    cities        = np.empty(total_rows, dtype=object)
    regions       = np.empty(total_rows, dtype=object)

    idx = 0
    for i, (_, row) in enumerate(df.iterrows()):
        duration = durations.iloc[i]
        if duration <= 0:
            continue

        end_idx = idx + duration

        # 10-min spaced timestamps across the session
        timestamps[idx:end_idx] = pd.date_range(
            start=row['Start Date'],
            periods=duration,
            freq='10min'
        ).values.astype('datetime64[m]')

        # Even energy split across the 10-min slots
        energy_per_interval = row['Total kWh'] / duration
        energy_values[idx:end_idx] = energy_per_interval

        # Copy session metadata to each expanded row
        station_names[idx:end_idx] = row['CP ID']
        cities[idx:end_idx]        = row['City']
        regions[idx:end_idx]       = row['Site']

        idx = end_idx

    # Build expanded per-interval DataFrame (trim to filled length)
    return pd.DataFrame({
        'Timestamp': pd.to_datetime(timestamps[:idx]),
        'Total kWh': energy_values[:idx],
        'CP ID': station_names[:idx],
        'City': cities[:idx],
        'Site': regions[:idx]
    })

df = expand_sessions_to_intervals(df)

# Align naming with other cities
df = df.rename(columns={"CP ID": "Station_Name"})
df['Station_Name'] = df['Station_Name'].astype(str)


# =========================================================
# MAKE DENSE 10-MIN GRID PER STATION
# =========================================================
def make_10min_grid_per_station(
    df: pd.DataFrame,
    target_in: str = 'Total kWh',
    target_out: str = 'Total kWh',
    region_col: str = 'Site'
) -> pd.DataFrame:
    """
    Sum overlapping intervals per (Station_Name, Region) into exact 10-min bins,
    enforce a dense grid per station lifespan, and one-hot encode station/region.

    Returns index=Timestamp with:
      ['Station_Name', 'Region', target_out, station one-hots, region one-hots]
    """
    df = df.copy()
    df['Timestamp'] = pd.to_datetime(df['Timestamp']).dt.floor('10min')
    df = df.drop(columns=['City'], errors='ignore')
    df['Region'] = df[region_col].astype(str)

    s_full = (
        df.groupby(['Station_Name', 'Region'], group_keys=True)
          .apply(lambda g: (
              g.set_index('Timestamp')
               .resample('10min')[target_in]
               .sum()
               .asfreq('10min', fill_value=0)
          ))
          .rename(target_out)
    )

    df_full = s_full.reset_index()

    dummies = pd.get_dummies(
        df_full[['Station_Name', 'Region']],
        prefix=['station', 'region'],
        dtype='uint8'
    )

    out = (
        pd.concat([df_full[['Timestamp', 'Station_Name', 'Region', target_out]], dummies], axis=1)
          .sort_values(['Timestamp', 'Station_Name'])
          .set_index('Timestamp')
    )
    return out

df_10min = make_10min_grid_per_station(df)

# Keep exact sort/index semantics from original (stable mergesort)
df_10min = (
    df_10min.reset_index()
            .sort_values(['Timestamp', 'Station_Name'], kind='mergesort')
            .set_index('Timestamp')
)


# =========================================================
# RESAMPLING (hourly / daily) PER STATION
# =========================================================
def resample_per_station(df_10min: pd.DataFrame, rule: str, target_col: str = 'Total kWh') -> pd.DataFrame:
    """
    Resample per (Station_Name, Region) to a coarser rule ('1h' or '1D'),
    summing energy per bin and keeping a dense grid, then add one-hots.

    Input:
      df_10min: index=Timestamp with ['Station_Name','Region', target_col] + one-hots
    """
    dfw = df_10min.reset_index()[['Timestamp', 'Station_Name', 'Region', target_col]].copy()

    parts = []
    for (st, reg), g in dfw.groupby(['Station_Name', 'Region'], sort=False):
        s = (g.set_index('Timestamp')[target_col]
               .resample(rule).sum()
               .asfreq(rule, fill_value=0))
        p = s.to_frame(name=target_col).reset_index()
        p['Station_Name'] = st
        p['Region'] = reg
        parts.append(p)

    base = pd.concat(parts, ignore_index=True)

    dummies = pd.get_dummies(
        base[['Station_Name', 'Region']],
        prefix=['station', 'region'],
        dtype='uint8'
    )

    out = (
        pd.concat([base[['Timestamp', 'Station_Name', 'Region', target_col]], dummies], axis=1)
          .sort_values(['Timestamp', 'Station_Name'])
          .set_index('Timestamp')
    )
    return out

# Build hourly and daily from the 10-minute grid (same logic/order as original)
hourly = resample_per_station(df_10min, '1h', target_col='Total kWh')
daily  = resample_per_station(df_10min, '1D', target_col='Total kWh')

# Drop raw identifiers (keep one-hots for modeling)
df_10min = df_10min.drop(columns=['Station_Name'], errors='ignore')
df_10min = df_10min.drop(columns=['Region'],       errors='ignore')

hourly = hourly.drop(columns=['Station_Name'], errors='ignore')
hourly = hourly.drop(columns=['Region'],       errors='ignore')

daily = daily.drop(columns=['Station_Name'], errors='ignore')
daily = daily.drop(columns=['Region'],       errors='ignore')


# =========================================================
# FEATURE ENGINEERING (calendar + station lags)
# =========================================================
def build_holidays(idx: pd.DatetimeIndex, us_holidays=None) -> pd.DatetimeIndex:
    """
    Return a DatetimeIndex of Western Australia holiday dates (normalized to midnight)
    covering the years present in idx. If precomputed holidays given, reuse.
    """
    if us_holidays is None:
        years = range(int(idx.year.min()), int(idx.year.max()) + 1)
        us_holidays = holidays.AU(state='WA', years=years)
    return pd.to_datetime(list(us_holidays.keys())).normalize()


def add_calendar_features(df: pd.DataFrame, us_holidays=None) -> pd.DataFrame:
    """
    Add calendar signals: is_holiday, hour, dayofweek, month.
    Keeps original columns; sets/uses DatetimeIndex.
    """
    df = df.copy()
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()
    else:
        df = df.set_index(pd.to_datetime(df['Timestamp'])).drop(columns='Timestamp').sort_index()

    idx = df.index
    holi_dates = build_holidays(idx, us_holidays)

    df['is_holiday'] = idx.normalize().isin(holi_dates).astype('uint8')
    df['hour']       = idx.hour.astype('uint8')       # 0 when resampled to daily
    df['dayofweek']  = idx.dayofweek.astype('uint8')  # 0=Mon
    df['month']      = idx.month.astype('uint8')
    return df


def add_station_lags(
    df: pd.DataFrame,
    target_col: str = 'Total kWh',
    station_prefix: str = 'station_',
    lags: list | None = None
) -> pd.DataFrame:
    """
    Create per-station lag features for the target.
    Uses one-hot station columns to build a stable station_id and applies groupby-shift.
    """
    if lags is None:
        lags = []

    df = df.copy().sort_index()

    station_cols = [c for c in df.columns if c.startswith(station_prefix)]
    if not station_cols:
        raise ValueError(f"No station one-hot columns found with prefix '{station_prefix}'")

    # Derive station identifier from one-hot
    df['_station_id'] = df[station_cols].idxmax(axis=1)

    g = df.groupby('_station_id', sort=False)[target_col]
    for L in lags:
        df[f'lag_{L}'] = g.shift(L)

    # Fill lag NaNs introduced by shift (no data before start)
    lag_mask = df.columns.str.startswith('lag_')
    df.loc[:, lag_mask] = df.loc[:, lag_mask].fillna(0.0)

    df.drop(columns=['_station_id'], inplace=True)
    return df


def make_features(
    df: pd.DataFrame,
    freq: str,
    us_holidays=None,
    station_prefix: str = 'station_'
) -> pd.DataFrame:
    """
    Add calendar features and station lags with sensible defaults for EV load.
    freq ∈ {'10min','1h','1D'} controls the lag set.
    """
    lag_map = {
        '10min': [1, 6, 12, 18, 24, 36, 72, 144, 288, 1008],  # 1 step, 1h, 2h, 3h, 4h, 6h, 12h, 1d, 2d, 1w
        '1h'   : [1, 2, 3, 6, 12, 24, 48, 72, 168],           # 1–3h, 6h, 12h, 1–3d, 1w
        '1D'   : [1, 2, 3, 7, 14, 28, 56, 91, 182, 365],      # 1–3d, weekly (1, 2, 4, 9, 13), 1/2 y, 1y
    }
    if freq not in lag_map:
        raise ValueError("freq must be one of {'10min','1h','1D'}")

    df_fe = add_calendar_features(df, us_holidays=us_holidays)
    df_fe = add_station_lags(
        df_fe,
        target_col='Total kWh',
        station_prefix=station_prefix,
        lags=lag_map[freq]
    )
    return df_fe


# Fixed holiday set (Western Australia), years as provided
us_holidays = holidays.AU(state='WA', years=range(2016, 2019))

df_10min = make_features(df_10min, freq='10min', us_holidays=us_holidays, station_prefix='station_')
df_hour  = make_features(hourly,   freq='1h',    us_holidays=us_holidays, station_prefix='station_')
df_day   = make_features(daily,    freq='1D',    us_holidays=us_holidays, station_prefix='station_')


# =========================================================
# SAVE RESULTS
# =========================================================
save_path = os.path.join('..', '..', 'data', 'data_preprocessed', 'preprocessed_perth')

with open(os.path.join(save_path, 'df_10min.pkl'), 'wb') as f:
    pickle.dump(df_10min, f)

with open(os.path.join(save_path, 'df_hour.pkl'), 'wb') as f:
    pickle.dump(df_hour, f)

with open(os.path.join(save_path, 'df_day.pkl'), 'wb') as f:
    pickle.dump(df_day, f)