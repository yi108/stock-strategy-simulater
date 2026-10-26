"""回測引擎：單一標的訊號回測、多標的自由組合、動量輪動，以及績效指標。

規則：
- 訊號以「當日收盤」計算，從「次日」開始持有（shift(1)），避免前視偏誤。
- 換手時收取雙邊交易成本 fee_bps + 滑價 slippage_bps（基點）。
- 停損/停利（sl_pct/tp_pct）以收盤價近似：當日收盤觸發，隔日出場生效，
  之後維持空手直到原始訊號改變（重新進場條件）。
- vol_target > 0 時做波動率目標倉位：倉位 × min(1, 目標年化波動/近20日年化波動)。
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

    @property
    def drawdown(self) -> pd.Series:
        return self.equity / self.equity.cummax() - 1.0

    def summary(self) -> dict:
        return {
            "total_return": round(self.total_return, 4),
            "cagr": round(self.cagr, 4),
            "sharpe": round(self.sharpe, 3),
            "max_drawdown": round(self.max_drawdown, 4),
            "trades": self.trades,
            **self.meta,
        }


def subresult(res: Result, lo: int | None = None, hi: int | None = None) -> Result:
    """取回測結果的一段（以報酬序列的位置切片），淨值重設為 1 起算。"""
    r = res.returns.iloc[lo:hi]
    eq = (1.0 + r).cumprod()
    return Result(equity=eq, returns=r, positions=r * 0, trades=res.trades)


def _apply_stops(eff: np.ndarray, close: pd.Series, sl: float, tp: float) -> np.ndarray:
    """停損/停利覆蓋層（收盤近似）。eff 為已 shift 的持倉陣列，就地修改後回傳。"""
    c = close.to_numpy()
    prev = 0.0
    entry = None
    blocked: float | None = None   # 被停損/停利後，原始訊號值不變期間維持空手
    for i in range(len(eff)):
        s = eff[i]
        if blocked is not None and s != blocked:
            blocked = None
        if blocked is not None:
            eff[i] = 0.0
            prev = 0.0
            continue
        if s == 0:
            entry = None
            prev = 0.0
            continue
        if prev == 0 or np.sign(s) != np.sign(prev):
            entry = c[i - 1] if i > 0 else c[i]
        pnl = (c[i] / entry - 1.0) * np.sign(s)
        if (sl > 0 and pnl <= -sl) or (tp > 0 and pnl >= tp):
            blocked = s
            entry = None
        prev = s
    return eff


def _extract_trades(pos: pd.Series, ret: pd.Series, dates: pd.Series) -> list[dict]:
    """從持倉序列切出每筆交易（進出場日、方向、區間報酬）。"""
    trades = []
    start = None
    side = 0.0
    p = pos.to_numpy()
    for i in range(len(p)):
        cur = np.sign(p[i])
        if start is None and cur != 0:
            start, side = i, cur
        elif start is not None and cur != side:
            seg = ret.iloc[start:i]
            trades.append({"entry": str(dates.iloc[start]), "exit": str(dates.iloc[i - 1]),
                           "side": "多" if side > 0 else "空", "days": i - start,
                           "ret": round(float((1 + seg).prod() - 1), 4)})
            start, side = (i, cur) if cur != 0 else (None, 0.0)
    if start is not None:
        seg = ret.iloc[start:]
        trades.append({"entry": str(dates.iloc[start]), "exit": str(dates.iloc[-1]),
                       "side": "多" if side > 0 else "空", "days": len(p) - start,
                       "ret": round(float((1 + seg).prod() - 1), 4), "open": True})
    return trades


def _run_simulated(df: pd.DataFrame, strategy: Strategy, fee_bps: float,
                   slippage_bps: float, cash_rate: float) -> Result:
    """自帶 simulate() 的策略（如逢跌加碼）：直接用它算出的淨值與持股比例。"""
    sim = strategy.simulate(df, fee_bps=fee_bps, slippage_bps=slippage_bps,
                            cash_rate=cash_rate)
    idx = pd.to_datetime(df["date"])
    equity = pd.Series(sim["equity"], index=idx)
    ret = equity.pct_change()
    ret.iloc[0] = equity.iloc[0] - 1.0
    pos = pd.Series(sim["exposure"], index=idx)
    return Result(equity=equity, returns=ret, positions=pos,
                  trades=len(sim["trades_log"]),
                  meta={"strategy": repr(strategy), "trades_log": sim["trades_log"]})


def run_signals(df: pd.DataFrame, strategy: Strategy, fee_bps: float = 10.0,
                slippage_bps: float = 0.0, vol_target: float = 0.0,
                cash_rate: float = 0.0) -> Result:
    """對單一標的執行訊號策略回測。cash_rate 為閒置現金的年利率（例如 0.04）。"""
    df = df.reset_index(drop=True)
    if hasattr(strategy, "simulate"):
        return _run_simulated(df, strategy, fee_bps, slippage_bps, cash_rate)
    close = df["close"]
    sig = strategy.signals(df).fillna(0.0).clip(-1, 1)
    eff = sig.shift(1).fillna(0.0).to_numpy(copy=True)   # 次日生效

    sl = float(strategy.params.get("sl_pct") or 0.0)
    tp = float(strategy.params.get("tp_pct") or 0.0)
    if sl > 0 or tp > 0:
        eff = _apply_stops(eff, close, sl, tp)
    pos = pd.Series(eff, index=close.index)

    asset_ret = close.pct_change().fillna(0.0)
    if vol_target and vol_target > 0:
        rv = (asset_ret.rolling(20).std() * np.sqrt(TRADING_DAYS)).shift(1)
        scale = (vol_target / rv).clip(upper=1.0).fillna(1.0)
        pos = pos * scale

    turnover = pos.diff().abs()
    turnover.iloc[0] = abs(pos.iloc[0]) if len(pos) else 0.0
    cost = turnover * (fee_bps + slippage_bps) / 10_000.0
    ret = pos * asset_ret - cost
    if np.any(np.asarray(cash_rate) != 0):
        ret = ret + (1.0 - pos.abs()).clip(lower=0.0) * np.asarray(cash_rate) / TRADING_DAYS
    equity = (1.0 + ret).cumprod()

    trades_log = _extract_trades(pos, ret, df["date"])
    idx = pd.to_datetime(df["date"])
    equity.index = ret.index = pos.index = idx
    trades = len(trades_log)   # 以實際進出場筆數計，避免 vol targeting 的每日微調灌水
    return Result(equity=equity, returns=ret, positions=pos, trades=trades,
                  meta={"strategy": repr(strategy), "trades_log": trades_log})


def run_constant_mix(df: pd.DataFrame, exposure: float, cash_rate=0.0) -> Result:
    """固定比例配置：每天維持 exposure 的資金在標的、其餘放現金（每日再平衡、不計成本）。

    用來當逢跌加碼的公平對照組：持股比例相同、但不挑時機。
    cash_rate 可為固定年利率，或與 df 等長的每日年利率序列。
    """
    df = df.reset_index(drop=True)
    r = df["close"].pct_change().fillna(0.0).to_numpy()
    n = len(r)
    rates = np.broadcast_to(np.asarray(cash_rate, dtype=float) / TRADING_DAYS, (n,))
    ret = exposure * r + (1.0 - exposure) * rates
    if n:
        ret[0] = 0.0
    idx = pd.to_datetime(df["date"])
    ret = pd.Series(ret, index=idx)
    equity = (1.0 + ret).cumprod()
    return Result(equity=equity, returns=ret, positions=pd.Series(exposure, index=idx))


def run_rows(universe: dict[str, pd.DataFrame],
             rows: list[tuple[str, Strategy, float]],
             fee_bps: float = 10.0, slippage_bps: float = 0.0,
             vol_target: float = 0.0, cash_rate: float = 0.0) -> Result:
    """自由組合回測：rows 為 (ticker, Strategy, weight) 列表。

    不以 ticker 為 key——同一檔股票可以重複出現、搭配不同策略與權重。
    """
    active = [(t, s, w) for (t, s, w) in rows if w > 0 and t in universe]
    if not active:
        raise ValueError("沒有有效的組合列（權重需 > 0 且標的需有資料）")
    total = sum(w for _, _, w in active)
    per = []
    trades = 0
    for t, s, w in active:
        r = run_signals(universe[t], s, fee_bps=fee_bps, slippage_bps=slippage_bps,
                        vol_target=vol_target, cash_rate=cash_rate)
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
        expo = r.positions.reindex(ret_df.index).abs()
        row_res = Result(equity=eq, returns=ret_df[i], positions=expo,
                         trades=r.trades,
                         meta={"trades_log": r.meta.get("trades_log", []),
                               "avg_exposure": round(float(expo.mean()), 4),
                               "end_exposure": round(float(expo.iloc[-1]), 4)})
        row_results.append({"ticker": t, "strategy": s, "weight": round(w, 4),
                            "result": row_res})
    return Result(equity=equity, returns=port_ret, positions=ret_df * 0, trades=trades,
                  meta={"rows": row_results})


def run_portfolio(universe: dict[str, pd.DataFrame],
                  strategies: dict[str, Strategy],
                  weights: dict[str, float],
                  fee_bps: float = 10.0) -> Result:
    """多標的組合回測（以 ticker 為 key 的舊介面，CLI 用）。"""
    active = {t: w for t, w in weights.items() if w > 0 and t in universe}
    if not active:
        raise ValueError("沒有有效的權重")
    rows = [(t, strategies[t], w) for t, w in active.items()]
    res = run_rows(universe, rows, fee_bps=fee_bps)
    res.meta = {"weights": {t: round(w / sum(active.values()), 3) for t, w in active.items()},
                "strategies": {t: repr(s) for t, s in strategies.items() if t in active}}
    return res


# ---------------------------------------------------------------- 動量輪動

def run_rotation(universe: dict[str, pd.DataFrame], lookback: int = 126,
                 top_n: int = 2, reb_days: int = 21, fee_bps: float = 10.0,
                 slippage_bps: float = 0.0) -> Result:
    """橫斷面動量輪動：每 reb_days 日，按過去 lookback 日報酬排名，
    等權買進最強的 top_n 檔，次日生效。"""
    from .data import close_matrix

    mat = close_matrix(universe)
    n_days, n_assets = mat.shape
    lookback = int(min(max(5, lookback), n_days - 2))
    top_n = int(min(max(1, top_n), n_assets))
    reb_days = int(max(1, reb_days))
    rets = mat.pct_change().fillna(0.0)

    w = pd.DataFrame(0.0, index=mat.index, columns=mat.columns)
    holdings_log = []
    i = lookback
    while i < n_days:
        mom = mat.iloc[i] / mat.iloc[i - lookback] - 1.0
        top = list(mom.sort_values(ascending=False).index[:top_n])
        nxt = min(i + reb_days, n_days)
        w.iloc[i + 1:nxt + 1] = 0.0
        for t in top:
            w.iloc[i + 1:nxt + 1, w.columns.get_loc(t)] = 1.0 / top_n
        holdings_log.append({"date": mat.index[i].strftime("%Y-%m-%d"), "hold": top})
        i = nxt

    turnover = w.diff().abs().sum(axis=1).fillna(0.0)
    cost = turnover * (fee_bps + slippage_bps) / 10_000.0
    port_ret = (w * rets).sum(axis=1) - cost
    port_ret = port_ret.iloc[lookback:]
    equity = (1.0 + port_ret).cumprod()
    trades = int((turnover > 1e-9).sum())
    return Result(equity=equity, returns=port_ret, positions=w, trades=trades,
                  meta={"rotation": {"lookback": lookback, "top_n": top_n,
                                     "reb_days": reb_days},
                        "holdings": holdings_log[-8:]})
