"""研究工具：逢跌加碼的多起點滾動測試，以及多種策略的「策略實驗室」比較。

從歷史資料中每隔 step 個交易日取一個起點，各自跑一段 window 天的策略
（每段都從全現金、以該段第一天為高點重新開始），並與兩個對照組比較：
- 買進持有：第一天全部買進
- 固定配置：持股比例等於該段逢跌加碼的平均持股，但不挑時機（每日再平衡）
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import run_constant_mix, run_signals
from .strategies import make


def pct_label(x: float) -> str:
    return f"{x * 100:g}%"


EXIT_LABELS = {0: "", 1: " + 回前高賣出"}


def dip_configs(triggers, buy_fracs, mode: int = 0, cooldown: int = 5,
                init_frac: float = 0.0, exit_mode: int = 0,
                tp_pct: float = 0.10) -> list[dict]:
    suffix = EXIT_LABELS.get(exit_mode, f" + 賺{pct_label(tp_pct)}賣出")
    return [{"label": f"跌{pct_label(t)}買{pct_label(f)}{suffix}", "dd_trigger": t,
             "buy_frac": f, "mode": mode, "cooldown": cooldown, "init_frac": init_frac,
             "exit_mode": exit_mode, "tp_pct": tp_pct}
            for f in buy_fracs for t in triggers]


def rolling_dip(df: pd.DataFrame, configs: list[dict], window: int = 252,
                step: int = 21, cash_rate=0.0, fee_bps: float = 10.0) -> pd.DataFrame:
    """回傳每個（起點 × 設定）一列的明細表。cash_rate 可為固定值或與 df 等長的序列。"""
    df = df.reset_index(drop=True)
    n = len(df)
    if n < window:
        raise ValueError(f"資料只有 {n} 個交易日，不夠一段 {window} 天的視窗")
    rates = np.broadcast_to(np.asarray(cash_rate, dtype=float), (n,))
    recs = []
    for s in range(0, n - window + 1, step):
        sub = df.iloc[s:s + window].reset_index(drop=True)
        cw = rates[s:s + window]
        bh = run_signals(sub, make("buyhold"), fee_bps=fee_bps)
        base = {"start": sub["date"].iloc[0], "end": sub["date"].iloc[-1],
                "bh_ret": bh.total_return, "bh_mdd": bh.max_drawdown,
                "cash_rate_avg": float(cw.mean())}
        for c in configs:
            params = {k: v for k, v in c.items() if k != "label"}
            res = run_signals(sub, make("dip", **params), fee_bps=fee_bps, cash_rate=cw)
            expo = float(res.positions.mean())
            mix = run_constant_mix(sub, expo, cw)
            recs.append({**base, "config": c["label"],
                         "dip_ret": res.total_return, "dip_mdd": res.max_drawdown,
                         "mix_ret": mix.total_return, "avg_expo": expo,
                         "end_expo": float(res.positions.iloc[-1]), "buys": res.trades})
    return pd.DataFrame(recs)


SPENT = 0.98   # 期末持股 ≥ 98% 視為資金已用盡


def summarize(win: pd.DataFrame) -> pd.DataFrame:
    """每個設定一列的統計；第一列為買進持有。"""
    first = win.drop_duplicates("start")
    rows = [{"config": "買進持有", "n": len(first),
             "mean": first["bh_ret"].mean(), "median": first["bh_ret"].median(),
             "worst": first["bh_ret"].min(), "best": first["bh_ret"].max(),
             "beat_bh": np.nan, "beat_mix": np.nan, "median_expo": 1.0,
             "worst_mdd": first["bh_mdd"].min(), "spent": np.nan}]
    for label, g in win.groupby("config", sort=False):
        rows.append({"config": label, "n": len(g),
                     "mean": g["dip_ret"].mean(), "median": g["dip_ret"].median(),
                     "worst": g["dip_ret"].min(), "best": g["dip_ret"].max(),
                     "beat_bh": (g["dip_ret"] > g["bh_ret"]).mean(),
                     "beat_mix": (g["dip_ret"] > g["mix_ret"]).mean(),
                     "median_expo": g["avg_expo"].median(),
                     "worst_mdd": g["dip_mdd"].min(),
                     "spent": (g["end_expo"] >= SPENT).mean()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 合成槓桿 ETF

def synth_leveraged(base: pd.DataFrame, cash: np.ndarray, leverage: float = 2.0,
                    fee: float = 0.0095, start_price: float = 100.0) -> pd.DataFrame:
    """用標的每日報酬合成每日再平衡的槓桿 ETF。

    每日報酬 = 槓桿 × 標的報酬 − (槓桿 − 1) × 短期利率 / 252 − 管理費 / 252
    （槓桿 ETF 借錢放大部位，要付接近短期利率的融資成本）。
    """
    r = base["close"].pct_change().fillna(0.0).to_numpy()
    daily = leverage * r - (leverage - 1.0) * cash / 252.0 - fee / 252.0
    daily[0] = 0.0
    close = start_price * np.cumprod(1.0 + np.maximum(daily, -0.999))
    return pd.DataFrame({"date": base["date"].to_numpy(), "open": close, "high": close,
                         "low": close, "close": close, "volume": 0})


def compare_series(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """比較兩條價格序列在重疊期間的年化報酬、最大回撤、日報酬相關性與年度差距。"""
    m = a[["date", "close"]].merge(b[["date", "close"]], on="date", suffixes=("_a", "_b"))
    out = {"start": m["date"].iloc[0], "end": m["date"].iloc[-1], "days": len(m)}
    for k in ("a", "b"):
        c = m[f"close_{k}"].to_numpy(dtype=float)
        out[f"cagr_{k}"] = float((c[-1] / c[0]) ** (252 / (len(c) - 1)) - 1.0)
        out[f"mdd_{k}"] = float((c / np.maximum.accumulate(c) - 1.0).min())
    ra, rb = m["close_a"].pct_change(), m["close_b"].pct_change()
    out["corr"] = float(ra.corr(rb))
    yr = m.assign(year=m["date"].str[:4]).groupby("year")
    yearly = yr.agg(a=("close_a", lambda s: s.iloc[-1] / s.iloc[0] - 1),
                    b=("close_b", lambda s: s.iloc[-1] / s.iloc[0] - 1))
    out["yearly_gap_median"] = float((yearly["a"] - yearly["b"]).abs().median())
    return out


# ---------------------------------------------------------------- 策略實驗室

STATIC_LEVELS = {0: (0.3, 0.5, 0.7), 1: (0.5,)}   # 第一檔標的 / 其他標的的固定持股比例


def _continuous_candidates(prices: dict[str, pd.DataFrame], cash: np.ndarray,
                           fee_bps: float, sma_days: int,
                           short_fee: float) -> dict[str, np.ndarray]:
    """在完整歷史上算出每日報酬的「持續型」策略：買進持有、固定比例、均線濾網、均線多空。"""
    out: dict[str, np.ndarray] = {}
    daily_cash = cash / 252.0
    for k, (t, df) in enumerate(prices.items()):
        r = df["close"].pct_change().fillna(0.0).to_numpy()
        out[f"{t} 買進持有"] = r
        for p in STATIC_LEVELS[min(k, 1)]:
            out[f"{t} {p:.0%} + 國庫券"] = p * r + (1 - p) * daily_cash
        trend = run_signals(df, make("sma", fast=1, slow=sma_days), fee_bps=fee_bps,
                            cash_rate=cash)
        out[f"{t} {sma_days}日均線濾網"] = trend.returns.to_numpy()
        ls = run_signals(df, make("sma", fast=1, slow=sma_days, short=1), fee_bps=fee_bps,
                         cash_rate=cash, short_fee=short_fee)
        out[f"{t} {sma_days}日均線多空"] = ls.returns.to_numpy()
    return out


def _window_stats(r: np.ndarray, s: int, window: int) -> tuple[float, float]:
    """從第 s 天收盤進場、持有到第 s+window-1 天收盤的報酬與期間最大回撤。"""
    eq = np.concatenate([[1.0], np.cumprod(1.0 + r[s + 1:s + window])])
    return float(eq[-1] - 1.0), float((eq / np.maximum.accumulate(eq) - 1.0).min())


def _dip_lab_configs() -> list[dict]:
    return (dip_configs([0.03], [0.05, 0.10])
            + dip_configs([0.03], [0.10], exit_mode=1)
            + dip_configs([0.03], [0.10], exit_mode=2, tp_pct=0.10))


def strategy_lab(prices: dict[str, pd.DataFrame], cash: np.ndarray, first_start: int,
                 window: int = 252, step: int = 21, fee_bps: float = 10.0,
                 sma_days: int = 200,
                 short_fee: float = 0.0095) -> tuple[pd.DataFrame, pd.DataFrame]:
    """比較多種候選策略在相同一年期滾動區間的表現。

    prices 需已對齊日期（data.align_universe），cash 為對齊的每日年利率。
    回傳 (每段明細, 各策略統計)。統計中的「全期」指從第一個起點持有到資料結束。
    """
    tickers = list(prices)
    n = len(next(iter(prices.values())))
    first_start = max(first_start, sma_days)
    starts = list(range(first_start, n - window + 1, step))
    if not starts:
        raise ValueError("資料不足，無法產生任何滾動區間（均線需要 "
                         f"{sma_days} 天暖機，每段 {window} 天）")
    cont = _continuous_candidates(prices, cash, fee_bps, sma_days, short_fee)
    dip_cfgs = _dip_lab_configs()

    recs = []
    for s in starts:
        start_date = prices[tickers[0]]["date"].iloc[s]
        for name, r in cont.items():
            ret, mdd = _window_stats(r, s, window)
            recs.append({"start": start_date, "strategy": name, "ret": ret, "mdd": mdd})
        cw = cash[s:s + window]
        for t in tickers:
            sub = prices[t].iloc[s:s + window].reset_index(drop=True)
            for c in dip_cfgs:
                params = {k: v for k, v in c.items() if k != "label"}
                res = run_signals(sub, make("dip", **params), fee_bps=fee_bps, cash_rate=cw)
                recs.append({"start": start_date, "strategy": f"{t} 逢跌加碼 {c['label']}",
                             "ret": res.total_return, "mdd": res.max_drawdown})
    win = pd.DataFrame(recs)

    full: dict[str, tuple[float, float]] = {}
    for name, r in cont.items():
        eq = np.cumprod(1.0 + r[first_start + 1:])
        full[name] = (float(eq[-1] ** (252 / len(eq)) - 1.0),
                      float((eq / np.maximum.accumulate(eq) - 1.0).min()))
    for t in tickers:
        sub = prices[t].iloc[first_start:].reset_index(drop=True)
        for c in dip_cfgs:
            params = {k: v for k, v in c.items() if k != "label"}
            res = run_signals(sub, make("dip", **params), fee_bps=fee_bps,
                              cash_rate=cash[first_start:])
            full[f"{t} 逢跌加碼 {c['label']}"] = (res.cagr, res.max_drawdown)

    base = win[win["strategy"] == f"{tickers[0]} 買進持有"].set_index("start")["ret"]
    rows = []
    for name, g in win.groupby("strategy", sort=False):
        g = g.set_index("start")
        full_cagr, full_mdd = full.get(name, (np.nan, np.nan))
        rows.append({"strategy": name, "mean": g["ret"].mean(), "median": g["ret"].median(),
                     "p10": g["ret"].quantile(0.10), "worst": g["ret"].min(),
                     "worst_mdd": g["mdd"].min(),
                     "beat_base": (g["ret"] > base.reindex(g.index)).mean(),
                     "full_cagr": full_cagr, "full_mdd": full_mdd})
    return win, pd.DataFrame(rows)


def pick_by_risk(summary: pd.DataFrame, budgets=(0.20, 0.35, 0.50, None)) -> list[dict]:
    """依「全期最大回撤」上限，挑出全期年化報酬最高的策略。"""
    picks = []
    for b in budgets:
        # 以畫面顯示的精度（0.01%）比較，避免 -20.00% 因浮點誤差被判為超過 -20%
        ok = summary if b is None else summary[summary["full_mdd"].round(4) >= -b]
        ok = ok.dropna(subset=["full_cagr"])
        if ok.empty:
            picks.append({"budget": b, "row": None})
            continue
        picks.append({"budget": b, "row": ok.sort_values(["full_cagr", "median"]).iloc[-1]})
    return picks


def by_start_year(win: pd.DataFrame, label: str) -> pd.DataFrame:
    g = win[win["config"] == label].copy()
    g["year"] = g["start"].str[:4]
    out = g.groupby("year").agg(
        n=("dip_ret", "size"), dip=("dip_ret", "median"), bh=("bh_ret", "median"),
        mix=("mix_ret", "median"),
        beat_bh=("dip_ret", lambda s: (s > g.loc[s.index, "bh_ret"]).mean()))
    return out.reset_index()
