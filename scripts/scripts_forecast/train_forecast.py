# train_forecast.py
# Trains ARIMA / XGBoost / GRU / LSTM / Transformer and saves values for metrics.py

from pathlib import Path
import pickle
import warnings
from typing import Optional, Tuple, Any

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

# XGBoost
import xgboost as xgb

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras import layers, Model


# =========================================================
# GPU / Mixed precision
# =========================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        try:
            # Avoid TF pre-allocating full GPU memory
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    mixed_precision.set_global_policy('mixed_float16')
    print(f"[TF] Using {len(gpus)} GPU(s) with mixed precision.")
else:
    print("[TF] No GPU detected; running on CPU.")


# =========================================================
# CONFIG
# =========================================================
PROJECT_DIR = Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd()

CITY_LIST  = ["perth", "dundee", "boulder", "palo_alto"]
MODEL_LIST = ["arima", "XGBoost", "GRU", "LSTM", "Transformer"]

K_STEPS = {"min": 3, "hour": 8, "day": 5}  # horizons for k-step from test start

TARGET = "Total kWh"
LAG_PREFIX = "lag_"
STATION_PREFIX = "station_"
REGION_PREFIX  = "region_"

# XGBoost params
XGB_PARAMS = {
    "tree_method": "hist",
    "device": "cuda",        # GPU
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.05,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
}
NUM_BOOST = 2000
ESR = 200

# TF training
BATCH_SIZE = 2048
EPOCHS = 200
PATIENCE = 20


# =========================================================
# DATA LOADER
# =========================================================
def load_data(city, project_dir=PROJECT_DIR):
    """
    Load preprocessed dataframes for a city at 10-min, hourly, and daily granularities.
    Accepts parquet/pkl/feather/csv. Ensures DatetimeIndex and sorted index.
    """
    base = project_dir / "data" / "data_preprocessed" / f"preprocessed_{city}"

    def read_any(stem):
        for name in [f"{stem}.parquet", f"{stem}.pkl", f"{stem}.feather", f"{stem}.csv"]:
            p = base / name
            if p.exists():
                if p.suffix == ".parquet":
                    df = pd.read_parquet(p)
                elif p.suffix == ".pkl":
                    with open(p, "rb") as f:
                        df = pickle.load(f)
                elif p.suffix == ".feather":
                    df = pd.read_feather(p)
                else:
                    df = pd.read_csv(p, parse_dates=["Timestamp"])
                # Ensure DatetimeIndex (fall back to 'Timestamp' column)
                if not isinstance(df.index, pd.DatetimeIndex):
                    if "Timestamp" in df.columns:
                        df = df.set_index(pd.to_datetime(df["Timestamp"])).drop(columns=["Timestamp"], errors="ignore")
                    else:
                        raise ValueError(f"{p} has no DatetimeIndex or Timestamp column.")
                return df.sort_index()
        raise FileNotFoundError(f"Could not find {stem} in {base}")

    df_min  = read_any("df_10min")
    df_hour = read_any("df_hour")
    df_day  = read_any("df_day")
    return df_min, df_hour, df_day


# =========================================================
# PREP: split + scale
# =========================================================
def split_and_scale(df, train_frac=0.65, val_frac=0.15, target=TARGET, lag_prefix=LAG_PREFIX):
    """
    Chronological split: train/val/test by fractions.
    Scale target + lag features with StandardScaler fitted on TRAIN ONLY.
    Also returns grouped feature lists (lags vs static) and one-hot groups.
    """
    idx = df.index.sort_values().unique()
    n = len(idx)
    t_train_end = idx[int(np.floor(train_frac*n)) - 1]
    t_val_end   = idx[int(np.floor((train_frac+val_frac)*n)) - 1]

    train = df.loc[df.index <= t_train_end].copy()
    val   = df.loc[(df.index > t_train_end) & (df.index <= t_val_end)].copy()
    test  = df.loc[df.index > t_val_end].copy()

    lag_cols = [c for c in df.columns if c.startswith(lag_prefix)]
    cols2scale = [c for c in [target] + lag_cols if c in train.columns]

    # Scale only train, apply to val/test for fair evaluation
    scaler = StandardScaler().fit(train[cols2scale].astype(np.float32))
    for part in (train, val, test):
        part[cols2scale] = scaler.transform(part[cols2scale].astype(np.float32))

    all_feats   = [c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c])]
    st_cols     = [c for c in all_feats if c.startswith(STATION_PREFIX)]
    rg_cols     = [c for c in all_feats if c.startswith(REGION_PREFIX)]
    lag_cols    = [c for c in all_feats if c.startswith(lag_prefix)]
    static_cols = [c for c in all_feats if c not in lag_cols]  # one-hots + calendar, etc.

    # Warn if one-hot stations appear only in TEST (unseen during training)
    if st_cols:
        seen = set(np.array(st_cols)[np.where((pd.concat([train[st_cols], val[st_cols]], axis=0) == 1))[1]])
        only_test = set(np.array(st_cols)[np.where((test[st_cols] == 1))[1]]) - seen
        if only_test:
            print("WARNING: Stations only in TEST (unseen):", sorted(only_test))

    return train, val, test, scaler, all_feats, st_cols, rg_cols, lag_cols, static_cols


# =========================================================
# ML MODELS (XGB + TF model builders)
# =========================================================
def train_xgb_gpu(train, val, feat_cols, target=TARGET):
    """
    Train an XGBoost regressor with early stopping on validation RMSE.
    Expects already-scaled target and features.
    """
    dtr = xgb.DMatrix(
        train[feat_cols].values.astype(np.float32),
        label=train[target].values.astype(np.float32),
        feature_names=feat_cols
    )
    dva = xgb.DMatrix(
        val[feat_cols].values.astype(np.float32),
        label=val[target].values.astype(np.float32),
        feature_names=feat_cols
    )
    booster = xgb.train(
        XGB_PARAMS,
        dtr,
        num_boost_round=NUM_BOOST,
        evals=[(dtr, "train"), (dva, "val")],
        early_stopping_rounds=ESR,
        verbose_eval=100
    )
    return booster

def build_gru(seq_len, static_dim, units=64):
    """GRU over lag sequence; optional static vector concatenated before final dense."""
    seq_in = layers.Input(shape=(seq_len, 1), name="seq")
    x = layers.GRU(units, dropout=0.1, recurrent_dropout=0.0)(seq_in)
    if static_dim > 0:
        stat_in = layers.Input(shape=(static_dim,), name="stat")
        h = layers.Concatenate()([x, stat_in])
        h = layers.Dense(64, activation="relu")(h)
        y = layers.Dense(1, dtype="float32")(h)
        return Model([seq_in, stat_in], y)
    y = layers.Dense(1, dtype="float32")(x)
    return Model(seq_in, y)

def build_lstm(seq_len, static_dim, units=64):
    """LSTM over lag sequence; optional static vector concatenated before final dense."""
    seq_in = layers.Input(shape=(seq_len, 1), name="seq")
    x = layers.LSTM(units, dropout=0.1, recurrent_dropout=0.0)(seq_in)
    if static_dim > 0:
        stat_in = layers.Input(shape=(static_dim,), name="stat")
        h = layers.Concatenate()([x, stat_in])
        h = layers.Dense(64, activation="relu")(h)
        y = layers.Dense(1, dtype="float32")(h)
        return Model([seq_in, stat_in], y)
    y = layers.Dense(1, dtype="float32")(x)
    return Model(seq_in, y)

def build_transformer(seq_len, static_dim,
                      d_model=128, num_heads=8, depth=4, ffn_dim=256, dropout=0.1):
    """
    Transformer encoder over lag sequence; optional static vector concatenated before head.
    NOTE: Identical to former build_transformer_full; only renamed by request.
    """
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
    seq_in = layers.Input(shape=(seq_len, 1), name="seq")

    # Project scalar lag at each step to d_model then add learnable positional embedding
    x = layers.Dense(d_model)(seq_in)
    pos_idx = tf.range(seq_len)[tf.newaxis, :]
    pos_emb = layers.Embedding(input_dim=seq_len, output_dim=d_model, name="pos_emb")(pos_idx)
    x = layers.Add()([x, pos_emb])

    # Stacked self-attention blocks
    for _ in range(depth):
        attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout)(x, x)
        x = layers.LayerNormalization(epsilon=1e-5)(x + layers.Dropout(dropout)(attn))
        f = layers.Dense(ffn_dim, activation="gelu")(x)
        f = layers.Dropout(dropout)(f)
        f = layers.Dense(d_model)(f)
        x = layers.LayerNormalization(epsilon=1e-5)(x + f)

    # Global pooling over sequence
    x = layers.GlobalAveragePooling1D()(x)

    # Optional static features
    if static_dim > 0:
        stat_in = layers.Input(shape=(static_dim,), name="stat")
        h = layers.Concatenate()([x, stat_in])
        h = layers.Dense(128, activation="relu")(h)
        y = layers.Dense(1, dtype="float32")(h)
        return Model([seq_in, stat_in], y)

    y = layers.Dense(1, dtype="float32")(x)
    return Model(seq_in, y)

def train_tf_model(model, train, val, lag_cols, static_cols, target=TARGET):
    """
    Compile + fit a TF model with EarlyStopping on val_loss.
    Converts frame to (seq, static, y) arrays using lag_cols/static_cols.
    """
    L = len(lag_cols)

    def to_arrays(df):
        # Reshape lag features to (batch, seq_len, 1)
        x_seq = df[lag_cols].values.astype(np.float32).reshape((-1, L, 1))
        x_stat = df[static_cols].values.astype(np.float32) if static_cols else None
        y = df[target].values.astype(np.float32)
        return x_seq, x_stat, y

    Xs_tr, Xst_tr, y_tr = to_arrays(train)
    Xs_va, Xst_va, y_va = to_arrays(val)

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    cbs = [tf.keras.callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True, monitor="val_loss")]

    # Two call-signatures depending on presence of static features
    if Xst_tr is None:
        model.fit(Xs_tr, y_tr, validation_data=(Xs_va, y_va),
                  epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2, callbacks=cbs)
    else:
        model.fit([Xs_tr, Xst_tr], y_tr, validation_data=([Xs_va, Xst_va], y_va),
                  epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2, callbacks=cbs)
    return model


# =========================================================
# FORECASTS (walk-forward + K-step, NaN-safe for TF)
# =========================================================
def forecast_walk_and_k(model_type, model, test_df, feature_cols, lag_cols, static_cols,
                        st_cols, rg_cols, k_steps=3, target=TARGET, lag_prefix=LAG_PREFIX):
    """
    Make predictions across the test set with:
      • Walk-forward (timestamp by timestamp), updating per-station state of recent predictions
      • K-step (first K timestamps) multi-horizon outputs for metrics

    Returns:
      preds_full: DataFrame indexed by Timestamp with ['station_id','region_id','y_pred_scaled','y_true_scaled']
      preds_k   : DataFrame with ['horizon','station_id','region_id','y_pred_scaled','y_true_scaled']
    """
    tmp = test_df.copy()
    # Derive station/region ids from one-hots when present
    tmp['_station_id'] = tmp[st_cols].idxmax(axis=1) if st_cols else "station_UNKNOWN"
    tmp['_region_id']  = tmp[rg_cols].idxmax(axis=1)  if rg_cols else "region_UNKNOWN"
    L = len(lag_cols)

    def _predict_batch(df_batch: pd.DataFrame) -> np.ndarray:
        """Branch to XGB or TF predict API."""
        if model_type == 'xgb':
            dmat = xgb.DMatrix(df_batch[feature_cols].values.astype(np.float32), feature_names=feature_cols)
            return model.predict(dmat).astype(np.float32)
        x_seq = df_batch[lag_cols].values.astype(np.float32).reshape((-1, L, 1)) if L > 0 else \
                np.zeros((len(df_batch), 0, 1), dtype=np.float32)
        if static_cols:
            x_stat = df_batch[static_cols].values.astype(np.float32)
            out = model.predict([x_seq, x_stat], verbose=0).ravel()
        else:
            out = model.predict(x_seq, verbose=0).ravel()
        return out.astype(np.float32)

    def _inject_from_state(frame: pd.DataFrame, state: dict, seen_mask: np.ndarray) -> pd.DataFrame:
        """
        For stations already seen in earlier timestamps, overwrite their lag columns with
        the rolling prediction 'state' so the sequence reflects model outputs. New stations
        (first appearance) keep original lags. NaNs are zero-filled as a last resort.
        """
        if L == 0:
            return frame.copy()
        f = frame.copy()
        sids = f['_station_id'].to_numpy()
        # Build matrix of per-station rolling lags (state) aligned to current batch order
        mat = np.vstack([state.get(sid, [np.nan]*L) for sid in sids]).astype(np.float32)
        for j in range(L):
            col = f'{lag_prefix}{j+1}'
            if col in f.columns:
                inj_vals = mat[:, j]
                new_col = f[col].to_numpy().astype(np.float32, copy=True)
                # Only inject for stations already seen previously in test
                new_col[seen_mask] = inj_vals[seen_mask]
                # Safety: replace NaNs that may still remain after injection
                nan_idx = np.isnan(new_col)
                if nan_idx.any():
                    new_col[nan_idx] = 0.0
                f.loc[:, col] = new_col
        return f

    # Initialize per-station rolling state with the first chronological row per station
    first_rows = (tmp.reset_index().sort_values('Timestamp').groupby('_station_id', as_index=False).first())
    state = {r['_station_id']: [r.get(f'{lag_prefix}{i+1}', np.nan) for i in range(L)] for _, r in first_rows.iterrows()}
    seen_stations = set()

    # ---------- Walk-forward over test timeline ----------
    preds_blocks = []
    for ts, frame in tmp.groupby(tmp.index, sort=True):
        sids = frame['_station_id'].to_numpy()
        # Stations that have been seen already in previous timestamps of test
        first_mask = np.array([sid not in seen_stations for sid in sids], dtype=bool)
        orig = frame.copy()
        inj  = _inject_from_state(frame, state, seen_mask=~first_mask)
        yhat = _predict_batch(inj)

        # Safeguard: if any preds NaN, recompute on original frame (no injected state)
        if np.isnan(yhat).any():
            bad = np.isnan(yhat)
            if bad.any():
                if model_type == 'xgb':
                    dmat_bad = xgb.DMatrix(orig.iloc[bad][feature_cols].values.astype(np.float32), feature_names=feature_cols)
                    yhat_bad = model.predict(dmat_bad).astype(np.float32)
                else:
                    x_seq_bad = orig.iloc[bad][lag_cols].values.astype(np.float32).reshape((-1, L, 1)) if L > 0 else \
                                np.zeros((bad.sum(), 0, 1), dtype=np.float32)
                    if static_cols:
                        x_stat_bad = orig.iloc[bad][static_cols].values.astype(np.float32)
                        yhat_bad = model.predict([x_seq_bad, x_stat_bad], verbose=0).ravel().astype(np.float32)
                    else:
                        yhat_bad = model.predict(x_seq_bad, verbose=0).ravel().astype(np.float32)
                yhat[bad] = yhat_bad
                yhat[np.isnan(yhat)] = 0.0

        preds_blocks.append(pd.DataFrame({
            'Timestamp': ts,
            'station_id': sids,
            'region_id':  frame['_region_id'].to_numpy(),
            'y_pred_scaled': yhat,
            'y_true_scaled': frame[target].values.astype(np.float32),
        }))

        # Update seen stations and roll state forward: prepend new pred then drop oldest
        seen_stations.update(sids.tolist())
        if L > 0:
            for sid, y in zip(sids, yhat):
                if not np.isfinite(y):
                    continue
                cur = state.get(sid, [np.nan]*L)
                state[sid] = [float(y)] + cur[:-1]

    preds_full = pd.concat(preds_blocks, ignore_index=True).set_index('Timestamp')

    # ---------- K-step horizons from the start of test ----------
    ts_sorted = np.array(sorted(tmp.index.unique()))
    k = int(min(max(k_steps, 0), max(0, len(ts_sorted) - 1)))
    if k == 0:
        preds_k = pd.DataFrame(columns=['horizon','station_id','region_id','y_pred_scaled','y_true_scaled'])
    else:
        frame0 = tmp.loc[ts_sorted[0]]
        if isinstance(frame0, pd.Series):
            frame0 = frame0.to_frame().T
        state = {r['_station_id']: [r.get(f'{lag_prefix}{i+1}', np.nan) for i in range(L)]
                 for _, r in frame0.reset_index(drop=True).iterrows()}
        out = []
        seen_stations = set()
        for h in range(1, k + 1):
            frame_h = tmp.loc[ts_sorted[h]]
            if isinstance(frame_h, pd.Series):
                frame_h = frame_h.to_frame().T
            sids_h = frame_h['_station_id'].to_numpy()
            first_mask_h = np.array([sid not in seen_stations for sid in sids_h], dtype=bool)
            orig_h = frame_h.copy()
            inj_h  = _inject_from_state(frame_h, state, seen_mask=~first_mask_h)
            yhat_h = _predict_batch(inj_h)

            # Same NaN fallback safety as above
            if np.isnan(yhat_h).any():
                bad = np.isnan(yhat_h)
                if bad.any():
                    if model_type == 'xgb':
                        dmat_bad = xgb.DMatrix(orig_h.iloc[bad][feature_cols].values.astype(np.float32), feature_names=feature_cols)
                        yhat_bad = model.predict(dmat_bad).astype(np.float32)
                    else:
                        x_seq_bad = orig_h.iloc[bad][lag_cols].values.astype(np.float32).reshape((-1, L, 1)) if L > 0 else \
                                    np.zeros((bad.sum(), 0, 1), dtype=np.float32)
                        if static_cols:
                            x_stat_bad = orig_h.iloc[bad][static_cols].values.astype(np.float32)
                            yhat_bad = model.predict([x_seq_bad, x_stat_bad], verbose=0).ravel().astype(np.float32)
                        else:
                            yhat_bad = model.predict(x_seq_bad, verbose=0).ravel().astype(np.float32)
                    yhat_h[bad] = yhat_bad
                    yhat_h[np.isnan(yhat_h)] = 0.0

            out.append(pd.DataFrame({
                'horizon': h,
                'station_id': sids_h,
                'region_id':  frame_h['_region_id'].to_numpy(),
                'y_pred_scaled': yhat_h,
                'y_true_scaled': frame_h[target].values.astype(np.float32),
            }))

            seen_stations.update(sids_h.tolist())
            if L > 0:
                for sid, y in zip(sids_h, yhat_h):
                    if not np.isfinite(y):
                        continue
                    cur = state.get(sid, [np.nan]*L)
                    state[sid] = [float(y)] + cur[:-1]

        preds_k = pd.concat(out, ignore_index=True)

    return preds_full, preds_k


# =========================================================
# RUNNERS (split/train/forecast wiring)
# =========================================================
def run_xgb(df: pd.DataFrame, k_steps: int):
    """Split/scale → train XGBoost → walk-forward + K-step forecast."""
    train, val, test, scaler, all_feats, st_cols, rg_cols, lag_cols, static_cols = split_and_scale(df)
    booster = train_xgb_gpu(train, val, feat_cols=all_feats, target=TARGET)
    preds_full, preds_k = forecast_walk_and_k(
        model_type='xgb',
        model=booster,
        test_df=test,
        feature_cols=all_feats,
        lag_cols=lag_cols,
        static_cols=static_cols,
        st_cols=st_cols,
        rg_cols=rg_cols,
        k_steps=k_steps,
        target=TARGET,
        lag_prefix=LAG_PREFIX,
    )
    return preds_full, preds_k

def run_tf(df: pd.DataFrame, k_steps: int, model_builder):
    """Split/scale → build (GRU/LSTM/Transformer) → train → walk-forward + K-step forecast."""
    train, val, test, scaler, all_feats, st_cols, rg_cols, lag_cols, static_cols = split_and_scale(df)
    seq_len = len(lag_cols)
    static_dim = len(static_cols)
    model = model_builder(seq_len=seq_len, static_dim=static_dim)
    model = train_tf_model(model, train, val, lag_cols, static_cols, target=TARGET)
    preds_full, preds_k = forecast_walk_and_k(
        model_type='tf',
        model=model,
        test_df=test,
        feature_cols=all_feats,
        lag_cols=lag_cols,
        static_cols=static_cols,
        st_cols=st_cols,
        rg_cols=rg_cols,
        k_steps=k_steps,
        target=TARGET,
        lag_prefix=LAG_PREFIX,
    )
    return preds_full, preds_k


# =========================================================
# STATISTICAL MODEL — ARIMA (per station)
# =========================================================
from statsmodels.tsa.arima.model import ARIMA as SM_ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning

# Silence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="Provided `endog` series has been differenced")
warnings.filterwarnings("ignore", message="A date index has been provided")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=ValueWarning)
warnings.filterwarnings("ignore", message="No frequency information")

CAP_POINTS     = {"10T": 2048, "H": 1536, "D": 730}  # cap train tail for speed
ARIMA_METHOD   = "statespace"
GRID_MAX_PQ    = (2, 2)
GRID_TRY_D     = (0, 1)
FALLBACK_ORDER = (1, 1, 0)
FIT_KW = dict(enforce_stationarity=True, enforce_invertibility=True)
MAXITER = 50

def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DatetimeIndex (or convert from 'Timestamp' column)."""
    if isinstance(df.index, pd.DatetimeIndex): return df.sort_index()
    if "Timestamp" in df.columns:
        out = df.copy(); out.index = pd.to_datetime(out["Timestamp"])
        return out.drop(columns=["Timestamp"], errors="ignore").sort_index()
    raise ValueError("ARIMA expects DatetimeIndex or 'Timestamp' column.")

def _ensure_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure station_id and region_id columns exist (derive from one-hots if needed)."""
    out = df.copy()
    if "station_id" not in out.columns:
        st = [c for c in out.columns if c.startswith(STATION_PREFIX)]
        out["station_id"] = out[st].idxmax(axis=1) if st else "station_UNKNOWN"
    if "region_id" not in out.columns:
        rg = [c for c in out.columns if c.startswith(REGION_PREFIX)]
        out["region_id"] = out[rg].idxmax(axis=1) if rg else "region_UNKNOWN"
    return out

def _split_cutoff(idx: pd.DatetimeIndex, train_frac=0.65):
    """Return the timestamp that separates train and (val+test)."""
    idx = idx.sort_values().unique()
    return idx[int(np.floor(train_frac*len(idx))) - 1]

def _infer_freq_str(idx: pd.DatetimeIndex) -> str:
    """Infer '10T' / 'H' / 'D' for capping by points."""
    if len(idx) < 2: return "D"
    step = (idx[1] - idx[0])
    m = int(round(step / pd.Timedelta(minutes=1)))
    if m == 10: return "10T"
    if m == 60: return "H"
    if m >= 1440: return "D"
    return f"{m}T"

def _cap_tail_by_points(y: pd.Series, cap: int) -> pd.Series:
    """Use recent tail of train series for faster gridsearch."""
    return y.iloc[-cap:] if len(y) > cap else y

def _fit_aic(y: pd.Series, order: Tuple[int, int, int]) -> Tuple[Optional[Any], float]:
    """Fit a candidate ARIMA(p,d,q); return (result, AIC)."""
    p, d, q = order
    if len(y) < (p + q + 8) or float(y.std()) < 1e-8:
        return None, np.inf
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = SM_ARIMA(y, order=order, trend="n", **FIT_KW).fit(
                method=ARIMA_METHOD,
                method_kwargs=dict(maxiter=MAXITER, disp=0),
            )
        aic = float(res.aic) if np.isfinite(res.aic) else np.inf
        return res, aic
    except Exception:
        return None, np.inf

def _select_order_fast(y: pd.Series, max_p=2, max_q=2, try_d=(0,1)) -> Tuple[Tuple[int, int, int], Optional[Any]]:
    """Tiny grid search (p,d,q) minimizing AIC; returns best order and its fitted model."""
    best_order, best_aic, best_res = FALLBACK_ORDER, np.inf, None
    for d in try_d:
        for p in range(max_p+1):
            for q in range(max_q+1):
                if p == 0 and d == 0 and q == 0:
                    continue
                res, aic = _fit_aic(y, (p,d,q))
                if aic < best_aic:
                    best_order, best_aic, best_res = (p,d,q), aic, res
    return best_order, best_res

def run_arima(df: pd.DataFrame, k_steps: int = 0) -> dict:
    """
    Per-station ARIMA on the scaled target.
    Trains on the TRAIN portion and forecasts the entire TEST period at once (no walk-forward).
    Returns dict with 'preds_full' DataFrame (Timestamp index): y_true_scaled, y_pred_scaled, station_id, region_id.
    """
    df = _ensure_dt_index(df)
    df = _ensure_ids(df)
    ycol = TARGET if TARGET in df.columns else ("Energy__kWh_" if "Energy__kWh_" in df.columns else None)
    if ycol is None:
        raise ValueError("Target column not found ('Total kWh' or 'Energy__kWh_').")

    t_train_end = _split_cutoff(df.index, 0.65)
    scaler = StandardScaler().fit(df.loc[df.index <= t_train_end, [ycol]].astype(np.float32))
    df["y_true_scaled"] = scaler.transform(df[[ycol]].astype(np.float32)).ravel()

    freq = _infer_freq_str(df.index)
    cap  = CAP_POINTS.get(freq, 2000)

    stations = list(df["station_id"].unique())
    print(f"[ARIMA-auto] stations={len(stations)}, freq={freq}, cap={cap}, grid p<={GRID_MAX_PQ[0]} q<={GRID_MAX_PQ[1]} d∈{list(GRID_TRY_D)}",
          flush=True)

    blocks = []
    for i, sid in enumerate(stations, start=1):
        g    = df[df["station_id"] == sid]
        y_tr = g.loc[g.index <= t_train_end, "y_true_scaled"].dropna()
        y_te = g.loc[g.index >  t_train_end, "y_true_scaled"]
        if y_te.empty:
            continue

        # Limit to recent tail for order selection
        y_tr = _cap_tail_by_points(y_tr, cap)
        print(f"[ARIMA-auto] {i}/{len(stations)} {sid} | train={len(y_tr):,} test={len(y_te):,}", flush=True)

        # Pick order by AIC on capped train tail
        order, res = _select_order_fast(y_tr, max_p=GRID_MAX_PQ[0], max_q=GRID_MAX_PQ[1], try_d=GRID_TRY_D)

        try:
            if res is None:
                res, _ = _fit_aic(y_tr, order)
            # Forecast the entire TEST span in one go
            yhat = np.asarray(res.forecast(steps=len(y_te)), dtype=np.float32) if res is not None else None
        except Exception:
            yhat = None

        if yhat is None:
            # Conservative fallback: last train value repeated
            last = float(y_tr.iloc[-1]) if len(y_tr) else 0.0
            yhat = np.full(shape=len(y_te), fill_value=last, dtype=np.float32)

        yhat = np.asarray(yhat, dtype=np.float32).ravel()
        y_te_vals = y_te.astype(np.float32).values
        n_te = len(y_te_vals)
        if len(yhat) != n_te:
            if len(yhat) > n_te:
                yhat = yhat[:n_te]
            else:
                pad_val = (yhat[-1] if len(yhat)
                           else (float(y_tr.iloc[-1]) if len(y_tr) else 0.0))
                yhat = np.pad(yhat, (0, n_te - len(yhat)), constant_values=pad_val)

        if "region_id" in g.columns and g["region_id"].notna().any():
            try:
                reg_value = g["region_id"].mode(dropna=True).iloc[0]
            except Exception:
                reg_value = g["region_id"].dropna().iloc[0]
        else:
            reg_value = "region_UNKNOWN"
        reg_vals = np.full(n_te, reg_value, dtype=object)

        blk = pd.DataFrame(
            {
                "y_true_scaled": y_te_vals,
                "y_pred_scaled": yhat,
                "station_id": sid,
                "region_id": reg_vals,
            },
            index=y_te.index,
        )
        blocks.append(blk)

    preds_full = (pd.concat(blocks, axis=0).sort_index()
                  if blocks else
                  pd.DataFrame(columns=["y_true_scaled","y_pred_scaled","station_id","region_id"]))
    return {"preds_full": preds_full}


# =========================================================
# SAVE RESULTS
# =========================================================
def save_results(city: str, model_name: str, results_by_granularity: dict, project_dir: Path = PROJECT_DIR):
    """
    Write one PKL per granularity with:
      index: DatetimeIndex
      cols : y_true_scaled, y_pred_scaled, station_id, region_id
    """
    def _unwrap_to_df(obj) -> pd.DataFrame:
        """Robustly unwrap nested results to the predictions DataFrame."""
        seen = 0
        while seen < 10:
            if isinstance(obj, pd.DataFrame):
                return obj
            if isinstance(obj, dict):
                for k in ("preds_full", "preds", "full", "predictions"):
                    if k in obj:
                        obj = obj[k]; break
                else:
                    if len(obj) == 1:
                        obj = next(iter(obj.values()))
                    else:
                        raise TypeError(f"Can't extract DataFrame from dict keys {list(obj.keys())}")
                seen += 1; continue
            if isinstance(obj, (tuple, list)):
                if not obj: raise TypeError("Empty tuple/list in results.")
                obj = obj[0]; seen += 1; continue
            raise TypeError(f"Unsupported result type while unwrapping: {type(obj)}")
        raise TypeError("Too many nesting levels while unwrapping results.")

    def _ensure_time_index(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure the predictions frame is indexed by Timestamp as DatetimeIndex."""
        if isinstance(df.index, pd.DatetimeIndex):
            return df
        if "Timestamp" in df.columns:
            out = df.copy()
            out.index = pd.to_datetime(out["Timestamp"])
            out.drop(columns=["Timestamp"], inplace=True, errors="ignore")
            return out
        if isinstance(df.index, pd.MultiIndex) and ("Timestamp" in (df.index.names or [])):
            ts = pd.to_datetime(df.index.get_level_values("Timestamp"))
            out = df.copy(); out.index = ts
            return out
        raise TypeError("Preds DataFrame must have a DatetimeIndex or a 'Timestamp' column.")

    def _ensure_ids(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure 'station_id' and 'region_id' columns exist."""
        out = df.copy()
        rename_map = {}
        if "_station_id" in out.columns and "station_id" not in out.columns:
            rename_map["_station_id"] = "station_id"
        if "_region_id" in out.columns and "region_id" not in out.columns:
            rename_map["_region_id"] = "region_id"
        if rename_map:
            out.rename(columns=rename_map, inplace=True)
        if "station_id" not in out.columns:
            st_cols = [c for c in out.columns if c.startswith("station_")]
            out["station_id"] = out[st_cols].idxmax(axis=1) if st_cols else "station_UNKNOWN"
        if "region_id" not in out.columns:
            rg_cols = [c for c in out.columns if c.startswith("region_")]
            out["region_id"] = out[rg_cols].idxmax(axis=1) if rg_cols else "region_UNKNOWN"
        return out

    base = project_dir / "results" / f"results_{city}" / f"{city}_{model_name}"
    base.mkdir(parents=True, exist_ok=True)

    for gran, res_obj in results_by_granularity.items():
        df = _unwrap_to_df(res_obj)
        df = _ensure_time_index(df)
        df = _ensure_ids(df)
        needed = ["y_true_scaled", "y_pred_scaled", "station_id", "region_id"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise KeyError(f"Missing columns for {city}/{model_name}/{gran}: {missing}")
        out = df[needed].copy()

        with open(base / f"{model_name}_{gran}.pkl", "wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)


# =========================================================
# Trainer wrappers
# =========================================================
def train_arima(df, k_steps):
    """ARIMA: per-station, target-only; forecasts the entire test range."""
    print("Training ARIMA (per-station) and forecasting test range")
    return run_arima(df, k_steps)
    
def train_xgboost(df, k_steps):
    """XGBoost: walk-forward over test + K-step horizons."""
    print(f"Training XGBoost and forecasting for {k_steps} steps")
    return run_xgb(df, k_steps)

def train_gru(df, k_steps):
    """GRU: walk-forward over test + K-step horizons."""
    print(f"Training GRU and forecasting for {k_steps} steps")
    return run_tf(df, k_steps, model_builder=build_gru)

def train_lstm(df, k_steps):
    """LSTM: walk-forward over test + K-step horizons."""
    print(f"Training LSTM and forecasting for {k_steps} steps")
    return run_tf(df, k_steps, model_builder=build_lstm)

def train_transformer(df, k_steps):
    """Transformer: walk-forward over test + K-step horizons."""
    print(f"Training Transformer and forecasting for {k_steps} steps")
    return run_tf(df, k_steps, model_builder=build_transformer)

TRAINERS = {
    "arima":       train_arima,
    "xgboost":     train_xgboost,
    "gru":         train_gru,
    "lstm":        train_lstm,
    "transformer": train_transformer,
}


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    for city in CITY_LIST:
        print(f"\n=== City: {city} ===")
        df_min, df_hour, df_day = load_data(city)

        for model_name in MODEL_LIST:
            print(f"\n-- Model: {model_name} --")
            trainer = TRAINERS[model_name]

            res_min  = trainer(df_min,  k_steps=K_STEPS["min"])
            res_hour = trainer(df_hour, k_steps=K_STEPS["hour"])
            res_day  = trainer(df_day,  k_steps=K_STEPS["day"])

            save_results(city, model_name, {"min": res_min, "hour": res_hour, "day": res_day})

    print("\nResults with values + IDs written for every city/model/granularity.")