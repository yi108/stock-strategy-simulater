"""資料層：抓取、記錄、載入每日報價。

CSV 格式統一為: date,open,high,low,close,volume（date 為 YYYY-MM-DD）。
每個 ticker 一個檔案，放在 data/ 目錄下，例如 data/AAPL.csv、data/2330.TW.csv。
"""

from __future__ import annotations

import io
import os
import urllib.request
from datetime import date, timedelta

import numpy as np
import pandas as pd

COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df[[c for c in COLUMNS if c in df.columns]]
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    # 指數（如 ^IRX）的成交量常為空值，只要求日期與收盤價存在
    df = df.dropna(subset=[c for c in ("date", "close") if c in df.columns])
    if "volume" in df.columns:
        df["volume"] = df["volume"].fillna(0)
    df = df.sort_values("date").drop_duplicates("date", keep="last")
    return df.reset_index(drop=True)


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _normalize(df)


def save_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    _normalize(df).to_csv(path, index=False)


def load_universe(data_dir: str, tickers: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """載入 data_dir 下的所有（或指定的）ticker CSV。"""
    out: dict[str, pd.DataFrame] = {}
    if tickers:
        names = [f"{t}.csv" for t in tickers]
    else:
        names = sorted(f for f in os.listdir(data_dir) if f.endswith(".csv"))
    for name in names:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            out[name[:-4]] = load_csv(path)
    return out


def close_matrix(universe: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """把多個 ticker 的收盤價合併成以日期為 index 的矩陣（取交集日期）。"""
    series = {}
    for t, df in universe.items():
        s = df.set_index("date")["close"]
        series[t] = s
    mat = pd.DataFrame(series).dropna()
    mat.index = pd.to_datetime(mat.index)
    return mat.sort_index()


def align_universe(universe: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """把多個 ticker 的日 K 對齊到共同交易日（交集），方便滾動切片。"""
    if not universe:
        return {}
    common = None
    for df in universe.values():
        s = set(df["date"])
        common = s if common is None else common & s
    common = sorted(common)
    out = {}
    for t, df in universe.items():
        sub = df[df["date"].isin(common)].sort_values("date").reset_index(drop=True)
        out[t] = sub
    return out


def slice_universe(universe: dict[str, pd.DataFrame],
                   lo: int | None = None, hi: int | None = None) -> dict[str, pd.DataFrame]:
    """對「已對齊」的 universe 以列位置切片（iloc 語意）。"""
    return {t: df.iloc[lo:hi].reset_index(drop=True) for t, df in universe.items()}


# ---------------------------------------------------------------- 真實資料抓取
# 注意：需要可以連外網的環境（例如你自己的電腦）。

def fetch_yfinance(ticker: str, period: str = "1y") -> pd.DataFrame:
    """用 yfinance 抓過去一段期間的日 K（需 pip install yfinance）。"""
    import yfinance as yf

    raw = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
    if raw.empty:
        return pd.DataFrame(columns=COLUMNS)
    df = raw.reset_index()
    df = df.rename(columns={"Date": "date", "Open": "open", "High": "high",
                            "Low": "low", "Close": "close", "Volume": "volume"})
    return _normalize(df)


def fetch_stooq(ticker: str, years: int = 1) -> pd.DataFrame:
    """從 stooq.com 抓日 K（美股代號加 .us，例如 aapl.us；台股不支援）。"""
    end = date.today()
    start = end - timedelta(days=int(365.25 * years))
    symbol = ticker.lower() if "." in ticker else f"{ticker.lower()}.us"
    url = (f"https://stooq.com/q/d/l/?s={symbol}&i=d"
           f"&d1={start:%Y%m%d}&d2={end:%Y%m%d}")
    with urllib.request.urlopen(url, timeout=30) as resp:
        text = resp.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(text))
    return _normalize(df)


class DataError(Exception):
    """抓不到或讀不到報價資料。"""


def fetch(ticker: str, period: str = "1y") -> pd.DataFrame:
    """先試 yfinance，失敗再試 stooq；兩者都失敗時拋出 DataError 並說明原因。"""
    errors = []
    try:
        df = fetch_yfinance(ticker, period=period)
        if len(df):
            return df
        errors.append("yfinance: 沒有回傳任何資料（代號可能錯誤）")
    except ImportError:
        errors.append("yfinance: 尚未安裝（請執行 pip install yfinance）")
    except Exception as e:
        errors.append(f"yfinance: {type(e).__name__}: {e}")
    try:
        years = max(1, int(period.rstrip("y") or 1)) if period.endswith("y") else 1
        df = fetch_stooq(ticker, years=years)
        if len(df):
            return df
        errors.append("stooq: 沒有回傳任何資料")
    except Exception as e:
        errors.append(f"stooq: {type(e).__name__}: {e}")
    raise DataError(f"抓不到 {ticker} 的報價：\n  " + "\n  ".join(errors))


def load_or_fetch(ticker: str, data_dir: str = "data", period: str = "1y",
                  log=print) -> pd.DataFrame:
    """讀取 data_dir/<ticker>.csv；檔案不存在時自動下載並存檔。"""
    path = os.path.join(data_dir, f"{ticker}.csv")
    if not os.path.exists(path):
        log(f"找不到 {path}，自動下載 {ticker} 最近 {period} 的報價…")
        df = fetch(ticker, period=period)
        save_csv(df, path)
        log(f"  已存檔 {path}（{len(df)} 根日K，{df['date'].iloc[0]} ~ {df['date'].iloc[-1]}）")
    return load_csv(path)


YF_PERIODS = {1, 2, 5, 10}


def load_history(ticker: str, data_dir: str = "data", years: int = 10,
                 name: str | None = None, log=print) -> pd.DataFrame:
    """確保 data_dir/<name>.csv 至少涵蓋最近 years 年；不夠就重新下載並合併存檔。"""
    path = os.path.join(data_dir, f"{name or ticker}.csv")
    need_start = (pd.Timestamp.today().normalize() - pd.DateOffset(years=years)
                  + pd.Timedelta(days=10))
    old = None
    if os.path.exists(path):
        old = load_csv(path)
        if len(old) and pd.Timestamp(old["date"].iloc[0]) <= need_start:
            return old
        log(f"{path} 只有 {old['date'].iloc[0]} 起的資料，下載 {ticker} 最近 {years} 年…")
    else:
        log(f"找不到 {path}，下載 {ticker} 最近 {years} 年…")
    period = f"{years}y" if years in YF_PERIODS else "max"
    new = fetch(ticker, period=period)
    merged = _normalize(pd.concat([old, new], ignore_index=True)) if old is not None else new
    save_csv(merged, path)
    log(f"  已存檔 {path}（{len(merged)} 根日K，{merged['date'].iloc[0]} ~ {merged['date'].iloc[-1]}）")
    if pd.Timestamp(merged["date"].iloc[0]) > need_start:
        log(f"  注意：{ticker} 最早只有 {merged['date'].iloc[0]} 的資料，不足 {years} 年")
    return merged


CASH_TICKER = "^IRX"      # 13 週美國國庫券殖利率（單位：%）
CASH_FILE = "CASH_IRX"


def load_cash_rates(dates, data_dir: str = "data", years: int = 10, log=print) -> np.ndarray:
    """取得與 dates 對齊的每日現金年利率（小數，例如 0.045），來源為 ^IRX。"""
    irx = load_history(CASH_TICKER, data_dir, years=years, name=CASH_FILE, log=log)
    s = irx.set_index("date")["close"].astype(float) / 100.0
    s.index = pd.to_datetime(s.index)
    idx = pd.to_datetime(pd.Series(dates))
    aligned = s.reindex(s.index.union(idx)).sort_index().ffill().bfill().reindex(idx)
    return aligned.clip(lower=0.0).to_numpy()


def record(ticker: str, data_dir: str = "data") -> pd.DataFrame:
    """記錄模式：抓最近的報價並 append 到既有 CSV（去重）。

    每天收盤後跑一次（可排 cron），一年後就累積出完整的未來一年資料。
    """
    path = os.path.join(data_dir, f"{ticker}.csv")
    new = fetch(ticker, period="5d")
    if os.path.exists(path):
        old = load_csv(path)
        merged = pd.concat([old, new], ignore_index=True)
    else:
        merged = new
    merged = _normalize(merged)
    save_csv(merged, path)
    return merged


# ---------------------------------------------------------------- 合成示範資料

def generate_synthetic(ticker_seed: int, days: int = 252, s0: float = 100.0,
                       mu: float = 0.08, sigma: float = 0.25) -> pd.DataFrame:
    """產生帶有趨勢/震盪區段的合成日 K，用於離線示範與測試。"""
    rng = np.random.default_rng(ticker_seed)
    dt = 1.0 / 252
    # 切成數個 regime，各自有不同漂移與波動，比單純 GBM 更像真實行情
    n_regimes = rng.integers(3, 6)
    bounds = np.sort(rng.choice(np.arange(20, days - 20), size=n_regimes - 1, replace=False))
    bounds = np.concatenate([[0], bounds, [days]])
    rets = np.empty(days)
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        m = mu * rng.uniform(-2.5, 3.5)
        v = sigma * rng.uniform(0.5, 1.8)
        rets[lo:hi] = (m - 0.5 * v * v) * dt + v * np.sqrt(dt) * rng.standard_normal(hi - lo)
    close = s0 * np.exp(np.cumsum(rets))
    spread = np.abs(rng.normal(0.006, 0.004, days))
    open_ = np.concatenate([[s0], close[:-1]]) * (1 + rng.normal(0, 0.003, days))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    volume = rng.integers(1_000_000, 20_000_000, days)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"), "open": open_.round(2),
        "high": high.round(2), "low": low.round(2),
        "close": close.round(2), "volume": volume,
    })


def generate_sample_universe(data_dir: str, tickers: list[str] | None = None,
                             days: int = 252) -> dict[str, pd.DataFrame]:
    tickers = tickers or ["DEMO_A", "DEMO_B", "DEMO_C", "DEMO_D", "DEMO_E"]
    out = {}
    for i, t in enumerate(tickers):
        df = generate_synthetic(ticker_seed=1000 + i * 7, days=days)
        save_csv(df, os.path.join(data_dir, f"{t}.csv"))
        out[t] = df
    return out
