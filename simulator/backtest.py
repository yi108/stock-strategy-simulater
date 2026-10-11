"""回測引擎：單一標的訊號回測 + 多標的投資組合回測，以及績效指標。

規則：
- 訊號以「當日收盤」計算，從「次日」開始持有（shift(1)），避免前視偏誤。
- 換手時收取雙邊交易成本 fee_bps（基點）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .strategies import Strategy

TRADING_DAYS = 252


@dataclass
class Result:
    equity: pd.Series                 # 淨值曲線（起始 1.0）
    returns: pd.Series                # 每日報酬
    positions: pd.Series | pd.DataFrame
    trades: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def total_return(self) -> float:
        return float(self.equity.iloc[-1] - 1.0)

    @property
    def cagr(self) -> float:
        n = len(self.returns)
        if n == 0 or self.equity.iloc[-1] <= 0:
            return float("nan")
        return float(self.equity.iloc[-1] ** (TRADING_DAYS / n) - 1.0)

    @property
    def sharpe(self) -> float:
        sd = self.returns.std()
        if not sd or np.isnan(sd) or sd == 0:
            return 0.0
        return float(self.returns.mean() / sd * np.sqrt(TRADING_DAYS))

    @property
    def max_drawdown(self) -> float:
        peak = self.equity.cummax()
        return float((self.equity / peak - 1.0).min())

    def summary(self) -> dict:
        return {
            "total_return": round(self.total_return, 4),
            "cagr": round(self.cagr, 4),
            "sharpe": round(self.sharpe, 3),
            "max_drawdown": round(self.max_drawdown, 4),
            "trades": self.trades,
            **self.meta,
        }


def run_signals(df: pd.DataFrame, strategy: Strategy, fee_bps: float = 10.0) -> Result:
    """對單一標的執行訊號策略回測。"""
    close = df["close"].reset_index(drop=True)
    sig = strategy.signals(df.reset_index(drop=True)).fillna(0.0).clip(0, 1)
    pos = sig.shift(1).fillna(0.0)          # 次日生效
    asset_ret = close.pct_change().fillna(0.0)
    turnover = pos.diff().abs().fillna(pos.iloc[0] if len(pos) else 0.0)
    cost = turnover * fee_bps / 10_000.0
    ret = pos * asset_ret - cost
    equity = (1.0 + ret).cumprod()
    idx = pd.to_datetime(df["date"]).reset_index(drop=True)
    equity.index = ret.index = pos.index = idx
    trades = int((turnover > 0).sum())
    return Result(equity=equity, returns=ret, positions=pos, trades=trades,
                  meta={"strategy": repr(strategy)})


def run_rows(universe: dict[str, pd.DataFrame],
             rows: list[tuple[str, Strategy, float]],
             fee_bps: float = 10.0) -> Result:
    """自由組合回測：rows 為 (ticker, Strategy, weight) 列表。

    與 run_portfolio 不同，這裡不以 ticker 為 key——同一檔股票可以
    重複出現、搭配不同策略與權重，完全自由排列。
    """
    active = [(t, s, w) for (t, s, w) in rows if w > 0 and t in universe]
    if not active:
        raise ValueError("沒有有效的組合列（權重需 > 0 且標的需有資料）")
    total = sum(w for _, _, w in active)
    per = []
    trades = 0
    for t, s, w in active:
        r = run_signals(universe[t], s, fee_bps=fee_bps)
        per.append((t, s, w / total, r))
        trades += r.trades
    ret_df = pd.DataFrame({i: p[3].returns for i, p in enumerate(per)}).dropna()
    if ret_df.empty:
        raise ValueError("各標的沒有重疊的交易日")
    port_ret = sum(ret_df[i] * p[2] for i, p in enumerate(per))
    equity = (1.0 + port_ret).cumprod()
    row_results = []
    for i, (t, s, w, r) in enumerate(per):
        eq = (1.0 + ret_df[i]).cumprod()
        row_res = Result(equity=eq, returns=ret_df[i], positions=ret_df[i] * 0,
                         trades=r.trades)
        row_results.append({"ticker": t, "strategy": s, "weight": round(w, 4),
                            "result": row_res})
    return Result(equity=equity, returns=port_ret, positions=ret_df * 0, trades=trades,
                  meta={"rows": row_results})


def run_portfolio(universe: dict[str, pd.DataFrame],
                  strategies: dict[str, Strategy],
                  weights: dict[str, float],
                  fee_bps: float = 10.0) -> Result:
    """多標的組合回測：各標的跑自己的策略，再按權重合成（每日再平衡的近似）。

    weights 不需要加總為 1；會自動正規化。權重為 0 的標的忽略。
    """
    active = {t: w for t, w in weights.items() if w > 0 and t in universe}
    if not active:
        raise ValueError("沒有有效的權重")
    total = sum(active.values())
    norm = {t: w / total for t, w in active.items()}

    rets = {}
    trades = 0
    for t, w in norm.items():
        res = run_signals(universe[t], strategies[t], fee_bps=fee_bps)
        rets[t] = res.returns
        trades += res.trades
    ret_df = pd.DataFrame(rets).dropna()
    port_ret = sum(ret_df[t] * w for t, w in norm.items())
    equity = (1.0 + port_ret).cumprod()
    return Result(equity=equity, returns=port_ret, positions=ret_df * 0, trades=trades,
                  meta={"weights": {t: round(w, 3) for t, w in norm.items()},
                        "strategies": {t: repr(s) for t, s in strategies.items()
                                       if t in norm}})
