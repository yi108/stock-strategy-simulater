"""技術指標：SMA、EMA、MACD、布林通道、RSI。皆以收盤價 Series 計算。"""

from __future__ import annotations

import pandas as pd


def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window).mean()


def ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """回傳 macd（DIF）、signal（DEA）、hist（柱狀圖）。"""
    dif = ema(close, fast) - ema(close, slow)
    dea = dif.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": dif, "signal": dea, "hist": dif - dea})


def bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """回傳 mid/upper/lower 與 %b（收盤價在通道中的相對位置）。"""
    mid = sma(close, window)
    std = close.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = (upper - lower).replace(0, pd.NA)
    pct_b = (close - lower) / width
    return pd.DataFrame({"mid": mid, "upper": upper, "lower": lower, "pct_b": pct_b})


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, pd.NA)
    return 100 - 100 / (1 + rs)
