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


# ---------------------------------------------------------------- 今日訊號

def trend_signal(df: pd.DataFrame, sma_days: int = 200) -> dict:
    """均線濾網在最新一根日K的訊號（與回測的 sma(fast=1, slow=sma_days) 規則相同）。

    回傳收盤、均線、距離、訊號、是否剛翻轉、目前狀態已持續幾天，以及
    「下一個交易日收盤要站上（或守住）均線的臨界價」：
    明天的均線 = (最近 sma_days-1 天收盤合計 + 明天收盤) / sma_days，
    明天收盤 > 明天均線 ⇔ 明天收盤 > 最近 sma_days-1 天收盤的平均。
    """
    close = df["close"].astype(float).reset_index(drop=True)
    if len(close) < sma_days + 1:
        raise ValueError(f"資料只有 {len(close)} 天，不夠計算 {sma_days} 日均線訊號")
    sma = close.rolling(sma_days).mean()
    above = (close > sma) & sma.notna()
    last = len(close) - 1
    state = bool(above.iloc[last])
    run = 0
    for v in above.iloc[::-1]:
        if bool(v) != state:
            break
        run += 1
    threshold = float(close.iloc[-(sma_days - 1):].mean())
    c = float(close.iloc[last])
    return {"date": str(df["date"].iloc[-1]), "close": c, "sma": float(sma.iloc[last]),
            "gap": c / float(sma.iloc[last]) - 1.0, "above": state,
            "switched": state != bool(above.iloc[last - 1]), "days_in_state": run,
            "threshold": threshold, "threshold_gap": threshold / c - 1.0}


def momentum_signal(df: pd.DataFrame, lookback: int = 120) -> dict:
    """時序動量濾網在最新一根日K的訊號（與回測的 tsmom(lookback) 規則相同）：
    今天收盤 > lookback 個交易日前的收盤 → 持有，否則國庫券。

    下一個交易日的臨界價 = 明天要比較的那一天（往回 lookback-1 天）的收盤價。
    """
    close = df["close"].astype(float).reset_index(drop=True)
    if len(close) < lookback + 2:
        raise ValueError(f"資料只有 {len(close)} 天，不夠計算 {lookback} 日動量訊號")
    mom = close / close.shift(lookback) - 1.0
    above = (mom > 0) & mom.notna()
    last = len(close) - 1
    state = bool(above.iloc[last])
    run = 0
    for v in above.iloc[::-1]:
        if bool(v) != state:
            break
        run += 1
    c = float(close.iloc[last])
    ref = float(close.iloc[last - lookback])
    threshold = float(close.iloc[last + 1 - lookback])
    return {"date": str(df["date"].iloc[-1]), "close": c, "ref": ref,
            "ref_date": str(df["date"].iloc[last - lookback]), "gap": c / ref - 1.0,
            "above": state, "switched": state != bool(above.iloc[last - 1]),
            "days_in_state": run, "threshold": threshold, "threshold_gap": threshold / c - 1.0}


def switch_returns(sig_df: pd.DataFrame, long_df: pd.DataFrame, alt_df: pd.DataFrame | None,
                   cash: np.ndarray, sma_days: int = 200, fee_bps: float = 10.0,
                   rule: str = "sma", lookback: int = 120,
                   alt_weight: float = 1.0) -> np.ndarray:
    """趨勢切換策略的每日報酬：sig_df 訊號為多 → 下一日持有 long_df，否則持有 alt_df。

    rule="sma"：收盤站上 sma_days 日均線；rule="mom"：收盤高於 lookback 天前收盤。
    alt_df 為 None 時改持有國庫券（等同濾網）。三者需已對齊日期。
    alt_weight < 1 時，避險部位只有該比例放 alt_df、其餘放國庫券（每日再平衡）。
    換手時兩邊都要成交（賣一檔、買另一檔），持有國庫券的部分只算一邊。
    """
    c = sig_df["close"].astype(float).reset_index(drop=True)
    ref = c.rolling(sma_days).mean() if rule == "sma" else c.shift(lookback)
    pos = ((c > ref) & ref.notna()).astype(float).shift(1).fillna(0.0).to_numpy()
    long_r = long_df["close"].pct_change().fillna(0.0).to_numpy()
    if alt_df is None:
        alt_r, legs = cash / 252.0, 1.0
    else:
        w = float(alt_weight)
        alt_r = (w * alt_df["close"].pct_change().fillna(0.0).to_numpy()
                 + (1.0 - w) * cash / 252.0)
        legs = 1.0 + w
    switches = np.abs(np.diff(pos, prepend=0.0))
    return pos * long_r + (1.0 - pos) * alt_r - switches * legs * fee_bps / 10_000.0


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
    """在完整歷史上算出每日報酬的「持續型」策略：買進持有、固定比例、均線濾網、
    120 日動量濾網（過去 120 天報酬為正持有，否則國庫券）、均線多空。"""
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
        mom = run_signals(df, make("tsmom", lookback=120), fee_bps=fee_bps, cash_rate=cash)
        out[f"{t} 120日動量濾網"] = mom.returns.to_numpy()
        ls = run_signals(df, make("sma", fast=1, slow=sma_days, short=1), fee_bps=fee_bps,
                         cash_rate=cash, short_fee=short_fee)
        out[f"{t} {sma_days}日均線多空"] = ls.returns.to_numpy()
    return out


def _window_stats(r: np.ndarray, s: int, window: int) -> tuple[float, float]:
    """從第 s 天收盤進場、持有到第 s+window-1 天收盤的報酬與期間最大回撤。"""
    eq = np.concatenate([[1.0], np.cumprod(1.0 + r[s + 1:s + window])])
    return float(eq[-1] - 1.0), float((eq / np.maximum.accumulate(eq) - 1.0).min())


def full_metrics(r: np.ndarray, rf_annual: np.ndarray) -> dict:
    """全期績效與風險溢價指標。r 為每日報酬，rf_annual 為同長度的每日國庫券年利率。

    風險溢價（超額年化）= 策略年化 − 國庫券年化；
    Sharpe = 每日超額報酬平均 / 每日報酬標準差 × √252；
    Sortino 只用下跌（超額報酬 < 0）的波動當分母；
    Calmar = 超額年化 / |最大回撤|。
    """
    r = np.asarray(r, dtype=float)
    rf = np.asarray(rf_annual, dtype=float) / 252.0
    years = len(r) / 252.0
    eq = np.cumprod(1.0 + r)
    cagr = float(eq[-1] ** (1.0 / years) - 1.0)
    mdd = float((eq / np.maximum.accumulate(eq) - 1.0).min())
    cash_cagr = float(np.prod(1.0 + rf) ** (1.0 / years) - 1.0)
    ex = r - rf
    sd = float(r.std(ddof=1))
    down = float(np.sqrt(np.mean(np.minimum(ex, 0.0) ** 2)))
    premium = cagr - cash_cagr
    return {"full_cagr": cagr, "full_mdd": mdd, "premium": premium,
            "vol": sd * np.sqrt(252.0),
            "sharpe": float(ex.mean() / sd * np.sqrt(252.0)) if sd > 0 else np.nan,
            "sortino": float(ex.mean() / down * np.sqrt(252.0)) if down > 0 else np.nan,
            "calmar": premium / abs(mdd) if mdd < 0 else np.nan}


def _dip_lab_configs() -> list[dict]:
    return (dip_configs([0.03], [0.05, 0.10])
            + dip_configs([0.03], [0.10], exit_mode=1)
            + dip_configs([0.03], [0.10], exit_mode=2, tp_pct=0.10))


def strategy_lab(prices: dict[str, pd.DataFrame], cash: np.ndarray, first_start: int,
                 window: int = 252, step: int = 21, fee_bps: float = 10.0,
                 sma_days: int = 200, short_fee: float = 0.0095,
                 alts: dict[str, pd.DataFrame] | None = None,
                 alt_weights: dict[str, float] | None = None
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    for t, df in prices.items():
        for a, adf in (alts or {}).items():
            w = (alt_weights or {}).get(a, 1.0)
            name = a.split(":")[0]
            label = name if w >= 1.0 else f"{w:.0%}{name}+國庫券"
            cont[f"{t} {sma_days}日均線 跌破換{label}"] = switch_returns(
                df, df, adf, cash, sma_days, fee_bps, alt_weight=w)
            cont[f"{t} 120日動量 轉負換{label}"] = switch_returns(
                df, df, adf, cash, sma_days, fee_bps, rule="mom", lookback=120, alt_weight=w)
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

    rf_full = cash[first_start + 1:]
    full: dict[str, dict] = {}
    for name, r in cont.items():
        full[name] = full_metrics(r[first_start + 1:], rf_full)
    for t in tickers:
        sub = prices[t].iloc[first_start:].reset_index(drop=True)
        for c in dip_cfgs:
            params = {k: v for k, v in c.items() if k != "label"}
            res = run_signals(sub, make("dip", **params), fee_bps=fee_bps,
                              cash_rate=cash[first_start:])
            full[f"{t} 逢跌加碼 {c['label']}"] = full_metrics(res.returns.to_numpy()[1:],
                                                             rf_full)

    base = win[win["strategy"] == f"{tickers[0]} 買進持有"].set_index("start")["ret"]
    rows = []
    for name, g in win.groupby("strategy", sort=False):
        g = g.set_index("start")
        rows.append({"strategy": name, "mean": g["ret"].mean(), "median": g["ret"].median(),
                     "p10": g["ret"].quantile(0.10), "worst": g["ret"].min(),
                     "worst_mdd": g["mdd"].min(),
                     "beat_base": (g["ret"] > base.reindex(g.index)).mean(),
                     **full[name]})
    return win, pd.DataFrame(rows)


def pick_by_risk(summary: pd.DataFrame, budgets=(0.20, 0.35, 0.50, None),
                 exclude: str = "逢跌加碼") -> list[dict]:
    """在「全期與任一年內的最大回撤」都不超過上限的策略中，挑出全期年化報酬最高者。

    同時看一年期回撤：全期只是一條路徑，從現金開始的一年期區間能補上不同進場時點的風險。
    名稱含 exclude 的策略不列入（逢跌加碼的全期數字取決於起點是否剛好空手）。
    """
    picks = []
    if exclude:
        summary = summary[~summary["strategy"].str.contains(exclude)]
    risk = summary[["full_mdd", "worst_mdd"]].min(axis=1)
    for b in budgets:
        # 以畫面顯示的精度（0.01%）比較，避免 -20.00% 因浮點誤差被判為超過 -20%
        ok = summary if b is None else summary[risk.round(4) >= -b]
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


# ---------------------------------------------------------------- 策略錦標賽（滾動最佳化）

def _score(r: np.ndarray, rf_annual: np.ndarray, objective: str) -> float:
    m = full_metrics(r, rf_annual)
    v = {"sharpe": m["sharpe"], "calmar": m["calmar"], "cagr": m["full_cagr"]}[objective]
    return v if np.isfinite(v) else -1e9


def walk_forward_strategy(df: pd.DataFrame, cash: np.ndarray, name: str, first_test: int,
                          test_days: int = 252, train_days: int = 756, warm: int = 400,
                          steps: int = 120, seed: int = 42, fee_bps: float = 10.0,
                          short_fee: float = 0.0095, objective: str = "sharpe"
                          ) -> tuple[np.ndarray, list[dict]]:
    """單一策略的滾動最佳化：每段只用「前 train_days 天」退火找參數，套用到接下來 test_days 天。

    參數空間含策略自身參數與共通的停損停利、做空開關（full_space）。
    每段以上一段的最佳參數為起點再退火；訓練與測試前都多給 warm 天讓指標暖機。
    回傳 (第 first_test 天起的樣本外每日報酬, 每段使用的參數)。
    """
    from .annealing import _perturb_param, anneal
    from .strategies import STRATEGIES

    cls = STRATEGIES[name]
    space = cls.full_space()
    n = len(df)
    oos = np.zeros(n)
    plog: list[dict] = []
    state = cls.full_defaults()

    def run(sub: pd.DataFrame, params: dict, c: np.ndarray) -> np.ndarray:
        return run_signals(sub, make(name, **params), fee_bps=fee_bps, cash_rate=c,
                           short_fee=short_fee).returns.to_numpy()

    for s in range(first_test, n, test_days):
        e = min(s + test_days, n)
        if space:
            lo = max(0, s - train_days)
            wlo = max(0, lo - warm)
            tr_df = df.iloc[wlo:s].reset_index(drop=True)
            tr_cash = cash[wlo:s]
            skip = lo - wlo + 1

            def score(st: dict) -> float:
                return _score(run(tr_df, st, tr_cash)[skip:], tr_cash[skip:], objective)

            def neighbor(st: dict, scale: float, rng) -> dict:
                nxt = dict(st)
                key = rng.choice(list(space))
                plo, phi, kind = space[key]
                nxt[key] = _perturb_param(st[key], plo, phi, kind, scale, rng)
                return nxt

            state = anneal(state, neighbor, score, steps=steps, seed=seed + s).best_state
        tlo = max(0, s - warm)
        r = run(df.iloc[tlo:e].reset_index(drop=True), state, cash[tlo:e])
        oos[s:e] = r[s - tlo:]
        plog.append({"start": str(df["date"].iloc[s]),
                     "params": {k: (round(v, 3) if isinstance(v, float) else v)
                                for k, v in state.items()}})
    return oos[first_test:], plog


def _segment_stats(r: np.ndarray, bh: np.ndarray, seg: int) -> dict:
    rets, mdds, beats = [], [], []
    for i in range(0, len(r), seg):
        eq = np.cumprod(1.0 + r[i:i + seg])
        rets.append(eq[-1] - 1.0)
        mdds.append((eq / np.maximum.accumulate(eq) - 1.0).min())
        # 與買進持有完全相同的報酬只會有浮點誤差，不算贏
        beats.append(rets[-1] > np.prod(1.0 + bh[i:i + seg]) - 1.0 + 1e-9)
    return {"median": float(np.median(rets)), "worst": float(np.min(rets)),
            "worst_mdd": float(np.min(mdds)), "beat_bh": float(np.mean(beats)),
            "segments": len(rets)}


def strategy_tournament(df: pd.DataFrame, cash: np.ndarray, first_test: int,
                        names: list[str] | None = None, test_days: int = 252,
                        train_days: int = 756, steps: int = 120, seed: int = 42,
                        fee_bps: float = 10.0, short_fee: float = 0.0095,
                        objective: str = "sharpe", sma_days: int = 200,
                        on_progress=None) -> tuple[pd.DataFrame, dict]:
    """所有策略在同一段樣本外期間比賽：預設參數版 vs 滾動最佳化版，並附買進持有與均線濾網。

    樣本外期間 = 第 first_test 天到最後；績效以 full_metrics（扣國庫券）與逐段（每 test_days 天）統計。
    回傳 (統計表, {策略: 最近一段使用的參數})。
    """
    from .strategies import STRATEGIES

    names = names or [k for k in STRATEGIES if k != "buyhold"]
    rf = cash[first_test:]
    bh = df["close"].pct_change().fillna(0.0).to_numpy()[first_test:]
    curves: dict[str, tuple[str, str, np.ndarray]] = {
        "買進持有": ("基準", "買進持有", bh),
        f"{sma_days}日均線濾網": ("基準", f"{sma_days}日均線濾網（固定規則）",
                                switch_returns(df, df, None, cash, sma_days, fee_bps)[first_test:]),
    }
    latest: dict[str, dict] = {}
    for i, name in enumerate(names):
        if on_progress:
            on_progress(i, len(names), name)
        d = run_signals(df, make(name), fee_bps=fee_bps, cash_rate=cash,
                        short_fee=short_fee).returns.to_numpy()[first_test:]
        curves[f"{name} 預設"] = (name, "預設參數", d)
        wf, plog = walk_forward_strategy(df, cash, name, first_test, test_days=test_days,
                                         train_days=train_days, steps=steps, seed=seed,
                                         fee_bps=fee_bps, short_fee=short_fee,
                                         objective=objective)
        curves[f"{name} 最佳化"] = (name, "滾動最佳化", wf)
        latest[name] = plog[-1]
    rows = []
    for key, (strat, version, r) in curves.items():
        rows.append({"strategy": key, "family": strat, "version": version,
                     **full_metrics(r, rf), **_segment_stats(r, bh, test_days)})
    return pd.DataFrame(rows), latest
