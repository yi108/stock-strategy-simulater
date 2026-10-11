"""模擬退火（Simulated Annealing）最佳化。

兩種用途：
1. optimize_params  — 對單一策略的參數空間做退火，找最佳指標參數（如 MACD 快慢線）。
2. optimize_portfolio — 同時對「各標的的策略參數 + 投資組合權重」做退火。

目標函數預設為 Sharpe，可改 total_return，或 Sharpe 減最大回撤懲罰（"calmar_mix"）。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from .backtest import Result, run_portfolio, run_signals
from .strategies import STRATEGIES, Strategy, make


def objective_value(res: Result, objective: str = "sharpe") -> float:
    if objective == "sharpe":
        return res.sharpe
    if objective == "total_return":
        return res.total_return
    if objective == "calmar_mix":
        return res.sharpe + 2.0 * res.max_drawdown  # max_drawdown 為負值，等同懲罰
    raise ValueError(f"未知目標函數: {objective}")


@dataclass
class AnnealLog:
    best_score: float = float("-inf")
    best_state: dict = field(default_factory=dict)
    history: list[tuple[int, float, float]] = field(default_factory=list)  # (step, temp, best)


def _perturb_param(value, lo, hi, kind, scale: float, rng: random.Random):
    span = (hi - lo) * scale
    if kind == "int":
        step = max(1, int(round(span)))
        nv = int(value) + rng.randint(-step, step)
        return int(min(hi, max(lo, nv)))
    nv = float(value) + rng.uniform(-span, span)
    return float(min(hi, max(lo, nv)))


def anneal(initial_state: dict,
           neighbor: Callable[[dict, float, random.Random], dict],
           score: Callable[[dict], float],
           steps: int = 400,
           t0: float = 1.0,
           t_end: float = 0.01,
           seed: int = 42,
           verbose: bool = False) -> AnnealLog:
    """通用模擬退火：指數降溫，Metropolis 接受準則。"""
    rng = random.Random(seed)
    state = dict(initial_state)
    current = score(state)
    log = AnnealLog(best_score=current, best_state=dict(state))
    alpha = (t_end / t0) ** (1.0 / max(1, steps - 1))
    temp = t0
    for step in range(steps):
        scale = 0.05 + 0.45 * (temp / t0)   # 高溫時跳得遠，低溫時微調
        cand = neighbor(state, scale, rng)
        cand_score = score(cand)
        delta = cand_score - current
        if delta >= 0 or rng.random() < math.exp(delta / max(temp, 1e-9)):
            state, current = cand, cand_score
        if current > log.best_score:
            log.best_score, log.best_state = current, dict(state)
        if step % 20 == 0:
            log.history.append((step, temp, log.best_score))
            if verbose:
                print(f"  step {step:4d}  T={temp:.4f}  current={current:+.3f}  best={log.best_score:+.3f}")
        temp *= alpha
    log.history.append((steps, temp, log.best_score))
    return log


# ---------------------------------------------------------------- 策略參數退火

def optimize_params(df: pd.DataFrame, strategy_name: str,
                    objective: str = "sharpe", fee_bps: float = 10.0,
                    steps: int = 400, seed: int = 42,
                    verbose: bool = False) -> tuple[Strategy, Result, AnnealLog]:
    cls = STRATEGIES[strategy_name]
    space = cls.PARAM_SPACE
    if not space:
        strat = cls()
        res = run_signals(df, strat, fee_bps=fee_bps)
        return strat, res, AnnealLog(best_score=objective_value(res, objective),
                                     best_state=dict(strat.params))

    def neighbor(state: dict, scale: float, rng: random.Random) -> dict:
        nxt = dict(state)
        key = rng.choice(list(space.keys()))
        lo, hi, kind = space[key]
        nxt[key] = _perturb_param(state[key], lo, hi, kind, scale, rng)
        return nxt

    def score(state: dict) -> float:
        res = run_signals(df, cls(**state), fee_bps=fee_bps)
        return objective_value(res, objective)

    log = anneal(cls.default_params(), neighbor, score, steps=steps, seed=seed, verbose=verbose)
    best = cls(**log.best_state)
    return best, run_signals(df, best, fee_bps=fee_bps), log


# ------------------------------------------------------- 組合（參數+權重）退火

def optimize_portfolio(universe: dict[str, pd.DataFrame], strategy_name: str,
                       objective: str = "sharpe", fee_bps: float = 10.0,
                       steps: int = 600, seed: int = 42,
                       verbose: bool = False) -> tuple[dict, Result, AnnealLog]:
    """對每個標的的策略參數與組合權重同時退火。

    狀態編碼: {"w:<ticker>": float 0..1, "p:<ticker>:<param>": value}
    """
    cls = STRATEGIES[strategy_name]
    space = cls.PARAM_SPACE
    tickers = list(universe.keys())

    init: dict = {}
    for t in tickers:
        init[f"w:{t}"] = 1.0 / len(tickers)
        for k, v in cls.default_params().items():
            init[f"p:{t}:{k}"] = v

    def decode(state: dict):
        weights = {t: max(0.0, state[f"w:{t}"]) for t in tickers}
        if sum(weights.values()) <= 0:
            weights = {t: 1.0 for t in tickers}
        strategies = {}
        for t in tickers:
            params = {k: state[f"p:{t}:{k}"] for k in cls.default_params()}
            strategies[t] = cls(**params)
        return weights, strategies

    def neighbor(state: dict, scale: float, rng: random.Random) -> dict:
        nxt = dict(state)
        key = rng.choice(list(state.keys()))
        if key.startswith("w:"):
            nv = state[key] + rng.uniform(-scale, scale)
            nxt[key] = min(1.0, max(0.0, nv))
        else:
            _, _t, pname = key.split(":", 2)
            lo, hi, kind = space[pname]
            nxt[key] = _perturb_param(state[key], lo, hi, kind, scale, rng)
        return nxt

    def score(state: dict) -> float:
        weights, strategies = decode(state)
        try:
            res = run_portfolio(universe, strategies, weights, fee_bps=fee_bps)
        except ValueError:
            return float("-inf")
        return objective_value(res, objective)

    log = anneal(init, neighbor, score, steps=steps, seed=seed, verbose=verbose)
    weights, strategies = decode(log.best_state)
    res = run_portfolio(universe, strategies, weights, fee_bps=fee_bps)
    total = sum(weights.values())
    best = {
        "weights": {t: round(w / total, 3) for t, w in weights.items() if w / total > 0.01},
        "params": {t: strategies[t].params for t in tickers},
    }
    return best, res, log
