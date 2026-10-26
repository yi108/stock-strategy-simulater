"""研究工具：逢跌加碼的多起點滾動測試。

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


def dip_configs(triggers, buy_fracs, mode: int = 0, cooldown: int = 5,
                init_frac: float = 0.0) -> list[dict]:
    return [{"label": f"跌{pct_label(t)}買{pct_label(f)}", "dd_trigger": t,
             "buy_frac": f, "mode": mode, "cooldown": cooldown, "init_frac": init_frac}
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


def by_start_year(win: pd.DataFrame, label: str) -> pd.DataFrame:
    g = win[win["config"] == label].copy()
    g["year"] = g["start"].str[:4]
    out = g.groupby("year").agg(
        n=("dip_ret", "size"), dip=("dip_ret", "median"), bh=("bh_ret", "median"),
        mix=("mix_ret", "median"),
        beat_bh=("dip_ret", lambda s: (s > g.loc[s.index, "bh_ret"]).mean()))
    return out.reset_index()
