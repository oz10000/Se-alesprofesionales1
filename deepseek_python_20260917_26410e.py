# indicators.py
# Ruta: indicators.py
# ============================================================
# D.A.P.S Ω — Indicadores técnicos (fórmulas Wilder)
# ============================================================

import numpy as np
import pandas as pd
from typing import Optional


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def _is_ok(df: Optional[pd.DataFrame]) -> bool:
    if df is None or df.empty:
        return False
    return all(c in df.columns for c in ("high", "low", "close", "volume"))


def _empty(series_index) -> pd.Series:
    return pd.Series(0.0, index=series_index)


# ------------------------------------------------------------
# TRUE RANGE / ATR
# ------------------------------------------------------------

def true_range(df: pd.DataFrame) -> pd.Series:
    if not _is_ok(df):
        return _empty(df.index if df is not None else None)
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.fillna(high - low)


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    if not _is_ok(df) or len(df) < period:
        return _empty(df.index if df is not None else None)
    tr = true_range(df)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    return atr.fillna(0).replace([np.inf, -np.inf], 0)


# ------------------------------------------------------------
# ADX WILDER
# ------------------------------------------------------------

def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    if not _is_ok(df) or len(df) < period * 2:
        return _empty(df.index if df is not None else None)

    high, low = df["high"], df["low"]
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    alpha = 1.0 / period
    tr = true_range(df)
    atr_s = tr.ewm(alpha=alpha, adjust=False).mean().replace(0, np.nan)

    plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_s
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_s

    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = (plus_di - minus_di).abs() / di_sum * 100
    dx = dx.fillna(0).replace([np.inf, -np.inf], 0)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    return adx.fillna(0).replace([np.inf, -np.inf], 0)


# ------------------------------------------------------------
# RSI WILDER
# ------------------------------------------------------------

def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    if not _is_ok(df) or len(df) < period:
        return _empty(df.index if df is not None else None).fillna(50.0)
    close = df["close"]
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50).replace([np.inf, -np.inf], 50).clip(0, 100)


# ------------------------------------------------------------
# EMA / SMA
# ------------------------------------------------------------

def compute_ema(df: pd.DataFrame, period: int) -> pd.Series:
    if df is None or df.empty or "close" not in df.columns:
        return _empty(df.index if df is not None else None)
    return df["close"].ewm(span=period, adjust=False).mean()


def compute_sma(df: pd.DataFrame, period: int) -> pd.Series:
    if df is None or df.empty or "close" not in df.columns:
        return _empty(df.index if df is not None else None)
    return df["close"].rolling(period).mean()


# ------------------------------------------------------------
# MACD
# ------------------------------------------------------------

def compute_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26,
                 signal: int = 9) -> pd.DataFrame:
    if df is None or df.empty or "close" not in df.columns:
        idx = df.index if df is not None else None
        return pd.DataFrame({"macd": 0.0, "signal": 0.0, "histogram": 0.0}, index=idx)
    close = df["close"]
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    macd = ema_f - ema_s
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return pd.DataFrame({
        "macd": macd.fillna(0),
        "signal": sig.fillna(0),
        "histogram": hist.fillna(0),
    })


# ------------------------------------------------------------
# ROC
# ------------------------------------------------------------

def compute_roc(df: pd.DataFrame, period: int = 10) -> pd.Series:
    if df is None or df.empty or "close" not in df.columns or len(df) < period:
        return _empty(df.index if df is not None else None)
    roc = (df["close"] / df["close"].shift(period) - 1) * 100
    return roc.fillna(0).replace([np.inf, -np.inf], 0).clip(-100, 100)


# ------------------------------------------------------------
# BOLLINGER
# ------------------------------------------------------------

def compute_bollinger(df: pd.DataFrame, period: int = 20,
                      std_mult: float = 2.0) -> pd.DataFrame:
    if df is None or df.empty or "close" not in df.columns or len(df) < period:
        idx = df.index if df is not None else None
        return pd.DataFrame({
            "bb_upper": 0.0, "bb_middle": 0.0,
            "bb_lower": 0.0, "bb_width": 0.0,
        }, index=idx)
    close = df["close"]
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    width = (upper - lower) / sma.replace(0, np.nan)
    return pd.DataFrame({
        "bb_upper": upper.fillna(close),
        "bb_middle": sma.fillna(close),
        "bb_lower": lower.fillna(close),
        "bb_width": width.fillna(0).replace([np.inf, -np.inf], 0),
    })


# ------------------------------------------------------------
# KAUFMAN EFFICIENCY RATIO
# ------------------------------------------------------------

def compute_ker(df: pd.DataFrame, period: int = 10) -> pd.Series:
    if df is None or df.empty or "close" not in df.columns or len(df) < period:
        return _empty(df.index if df is not None else None)
    close = df["close"]
    change = close.diff(period).abs()
    volatility = close.diff().abs().rolling(period).sum()
    ker = change / (volatility + 1e-9)
    return ker.fillna(0).replace([np.inf, -np.inf], 0).clip(0, 1)


# ------------------------------------------------------------
# VOLUMEN INTELIGENTE
# ------------------------------------------------------------

def compute_volume_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
    if df is None or df.empty or "volume" not in df.columns:
        return _empty(df.index if df is not None else None).fillna(1.0)
    avg = df["volume"].rolling(period).mean().replace(0, np.nan)
    vr = df["volume"] / avg
    return vr.fillna(1.0).replace([np.inf, -np.inf], 1.0).clip(0, 20)


def compute_delta_volume(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return _empty(df.index if df is not None else None)
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    pos = (df["close"] - df["low"]) / rng
    delta = df["volume"] * (2 * pos - 1)
    return delta.fillna(0).replace([np.inf, -np.inf], 0)


def compute_cvd(df: pd.DataFrame, period: int = 20) -> pd.Series:
    if df is None or df.empty:
        return _empty(df.index if df is not None else None)
    delta = compute_delta_volume(df)
    cvd = delta.rolling(period).sum()
    return cvd.fillna(0).replace([np.inf, -np.inf], 0)


def compute_tfi(df: pd.DataFrame, period: int = 5) -> pd.Series:
    """Trade Flow Imbalance aproximado."""
    if df is None or df.empty:
        return _empty(df.index if df is not None else None)
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    buy = (df["close"] - df["low"]) / rng
    buy = buy.clip(0, 1).fillna(0.5)
    buy_vol = df["volume"] * buy
    sell_vol = df["volume"] * (1 - buy)
    tfi = (buy_vol - sell_vol).rolling(period).sum()
    total = df["volume"].rolling(period).sum().replace(0, np.nan)
    tfi_norm = tfi / total
    return tfi_norm.fillna(0).replace([np.inf, -np.inf], 0).clip(-1, 1)


def compute_ofi(df: pd.DataFrame, period: int = 5) -> pd.Series:
    """Order Flow Imbalance aproximado."""
    if df is None or df.empty:
        return _empty(df.index if df is not None else None)
    dp = df["close"].diff().fillna(0)
    vw = dp * df["volume"]
    ofi = vw.rolling(period).sum()
    denom = (df["close"].abs().rolling(period).sum()
             * df["volume"].rolling(period).sum()).replace(0, np.nan)
    ofi_norm = ofi / denom
    return ofi_norm.fillna(0).replace([np.inf, -np.inf], 0).clip(-1, 1)


# ------------------------------------------------------------
# RÉGIMEN
# ------------------------------------------------------------

def compute_regime(adx: float, atr_pct: float) -> str:
    if adx != adx or atr_pct != atr_pct:
        return "Chop"
    if adx > 40 and atr_pct > 0.02:
        return "Expansion"
    if adx > 30:
        return "Trend_Strong"
    if adx > 20:
        return "Trend_Weak"
    return "Chop"