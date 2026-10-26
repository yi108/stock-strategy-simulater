"""模擬退火（Simulated Annealing）最佳化與驗證工具。

- optimize_params     單一策略參數退火（CLI 用）
- optimize_rows       自由組合（各列參數 + 權重）退火，支援訓練/驗證切分
- optimize_rotation   動量輪動參數退火
- walk_forward        滾動視窗：逐段退火 → 下一段樣本外驗證，拼出 OOS 淨值
- scan_params         參數穩健性網格掃描（熱圖用）

共通能力：
- restarts 多重啟（不同種子跑多次取最佳，避免局部最優）
- on_step 進度回報、should_stop 隨時中止（回傳目前最佳解）
- train_frac < 1 時，目標函數只在前段（訓練期）評分
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .backtest import Result, run_rotation, run_rows, run_signals, subresult
from .strategies import STRATEGIES, Strategy

ProgressFn = Callable[[int, int, float], None]       # (done, total, best)
StopFn = Callable[[], bool]


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
    stopped: bool = False


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
           restarts: int = 1,
           on_step: ProgressFn | None = None,
           should_stop: StopFn | None = None,
           verbose: bool = False) -> AnnealLog:
    """通用模擬退火：指數降溫，Metropolis 接受準則，可多重啟/回報進度/中止。"""
    restarts = max(1, int(restarts))
    total = steps * restarts
    log = AnnealLog()
    alpha = (t_end / t0) ** (1.0 / max(1, steps - 1))
    init_score = score(initial_state)
    if init_score > log.best_score:
        log.best_score, log.best_state = init_score, dict(initial_state)

    done = 0
    for r in range(restarts):
        rng = random.Random(seed + 1013 * r)
        # 第一輪從使用者初始狀態出發；之後從目前全域最佳出發再退火（re-anneal）
        state = dict(initial_state if r == 0 else log.best_state)
        current = score(state)
        temp = t0
        for step in range(steps):
            if should_stop and should_stop():
                log.stopped = True
                log.history.append((done, temp, log.best_score))
                return log
            scale = 0.05 + 0.45 * (temp / t0)   # 高溫跳得遠，低溫微調
            cand = neighbor(state, scale, rng)
            cand_score = score(cand)
            delta = cand_score - current
            if delta >= 0 or rng.random() < math.exp(delta / max(temp, 1e-9)):
                state, current = cand, cand_score
            if current > log.best_score:
                log.best_score, log.best_state = current, dict(state)
            if step % 10 == 0:
                log.history.append((done, temp, log.best_score))
                if on_step:
                    on_step(done, total, log.best_score)
                if verbose:
                    print(f"  step {done:5d}/{total}  T={temp:.4f}  best={log.best_score:+.3f}")
            temp *= alpha
            done += 1
    log.history.append((done, temp, log.best_score))
    if on_step:
        on_step(done, total, log.best_score)
    return log


# ---------------------------------------------------------------- 策略參數退火

def optimize_params(df: pd.DataFrame, strategy_name: str,
                    objective: str = "sharpe", fee_bps: float = 10.0,
                    slippage_bps: float = 0.0, train_frac: float = 1.0,
                    steps: int = 400, restarts: int = 1, seed: int = 42,
                    verbose: bool = False) -> tuple[Strategy, Result, AnnealLog]:
    cls = STRATEGIES[strategy_name]
    space = cls.full_space()
    n = len(df)
    n_train = int(n * train_frac) if train_frac < 1.0 else None

    def score(state: dict) -> float:
        res = run_signals(df, cls(**state), fee_bps=fee_bps, slippage_bps=slippage_bps)
        return objective_value(res if n_train is None else subresult(res, 0, n_train),
                               objective)

    if not space:
        strat = cls()
        res = run_signals(df, strat, fee_bps=fee_bps, slippage_bps=slippage_bps)
        return strat, res, AnnealLog(best_score=objective_value(res, objective),
                                     best_state=dict(strat.params))

    def neighbor(state: dict, scale: float, rng: random.Random) -> dict:
        nxt = dict(state)
        key = rng.choice(list(space.keys()))
        lo, hi, kind = space[key]
        nxt[key] = _perturb_param(state[key], lo, hi, kind, scale, rng)
        return nxt

    log = anneal(cls.full_defaults(), neighbor, score, steps=steps, seed=seed,
                 restarts=restarts, verbose=verbose)
    best = cls(**log.best_state)
    return best, run_signals(df, best, fee_bps=fee_bps, slippage_bps=slippage_bps), log


# ------------------------------------------------- 自由組合（任意列）退火

def optimize_rows(universe: dict[str, pd.DataFrame], specs: list[dict],
                  opt_weights: bool = True, opt_params: bool = True,
                  objective: str = "sharpe", fee_bps: float = 10.0,
                  slippage_bps: float = 0.0, vol_target: float = 0.0,
                  cash_rate: float = 0.0,
                  train_frac: float = 1.0, eval_slice: tuple[int, int] | None = None,
                  steps: int = 600, restarts: int = 1, seed: int = 42,
                  on_step: ProgressFn | None = None, should_stop: StopFn | None = None,
                  verbose: bool = False) -> tuple[list[dict], Result, AnnealLog]:
    """對使用者自由排列的組合列做退火。

    specs: [{"ticker", "strategy", "params": {...}, "weight",
             "opt_params": bool（選填，預設 True）}]
    目標函數評分範圍：eval_slice=(lo,hi) 指定報酬序列切片；
    否則 train_frac<1 時取前 train_frac 段（訓練期）。
    """
    classes = [STRATEGIES[s["strategy"]] for s in specs]

    init: dict = {}
    mutable: list[str] = []
    for i, (spec, cls) in enumerate(zip(specs, classes)):
        wkey = f"w:{i}"
        init[wkey] = float(spec.get("weight", 1.0))
        if opt_weights:
            mutable.append(wkey)
        defaults = cls.full_defaults()
        space = cls.full_space()
        user_params = spec.get("params") or {}
        row_opt = opt_params and spec.get("opt_params", True)
        for k, dv in defaults.items():
            pkey = f"p:{i}:{k}"
            init[pkey] = user_params.get(k, dv)
            if row_opt and k in space:
                mutable.append(pkey)

    def decode(state: dict) -> list[tuple[str, Strategy, float]]:
        rows = []
        for i, (spec, cls) in enumerate(zip(specs, classes)):
            params = {k: state[f"p:{i}:{k}"] for k in cls.full_defaults()}
            rows.append((spec["ticker"], cls(**params), max(0.0, state[f"w:{i}"])))
        return rows

    run = lambda state: run_rows(universe, decode(state), fee_bps=fee_bps,
                                 slippage_bps=slippage_bps, vol_target=vol_target,
                                 cash_rate=cash_rate)

    first = run(init)
    n = len(first.returns)
    if eval_slice is not None:
        lo, hi = eval_slice
    elif train_frac < 1.0:
        lo, hi = 0, int(n * train_frac)
    else:
        lo, hi = None, None

    def score(state: dict) -> float:
        try:
            res = run(state)
        except ValueError:
            return float("-inf")
        if lo is not None:
            res = subresult(res, lo, hi)
        return objective_value(res, objective)

    if not mutable:  # 沒東西可調，直接回測
        log = AnnealLog(best_score=score(init), best_state=dict(init))
    else:
        def neighbor(state: dict, scale: float, rng: random.Random) -> dict:
            nxt = dict(state)
            key = rng.choice(mutable)
            if key.startswith("w:"):
                nxt[key] = min(1.0, max(0.0, state[key] + rng.uniform(-scale, scale)))
            else:
                _, i, pname = key.split(":", 2)
                plo, phi, kind = classes[int(i)].full_space()[pname]
                nxt[key] = _perturb_param(state[key], plo, phi, kind, scale, rng)
            return nxt

        log = anneal(init, neighbor, score, steps=steps, seed=seed, restarts=restarts,
                     on_step=on_step, should_stop=should_stop, verbose=verbose)

    best_rows = decode(log.best_state)
    res = run_rows(universe, best_rows, fee_bps=fee_bps,
                   slippage_bps=slippage_bps, vol_target=vol_target, cash_rate=cash_rate)
    total = sum(w for _, _, w in best_rows) or 1.0
    best_specs = []
    for (t, strat, w), spec in zip(best_rows, specs):
        best_specs.append({
            "ticker": t, "strategy": strat.NAME,
            "params": {k: (round(v, 3) if isinstance(v, float) else v)
                       for k, v in strat.params.items()},
            "weight": round(w / total, 3),
            "opt_params": spec.get("opt_params", True),
        })
    return best_specs, res, log


# ---------------------------------------------------------------- 動量輪動退火

def optimize_rotation(universe: dict[str, pd.DataFrame], params: dict | None = None,
                      objective: str = "sharpe", fee_bps: float = 10.0,
                      slippage_bps: float = 0.0, train_frac: float = 1.0,
                      steps: int = 300, restarts: int = 1, seed: int = 42,
                      on_step: ProgressFn | None = None,
                      should_stop: StopFn | None = None) -> tuple[dict, Result, AnnealLog]:
    from .data import close_matrix

    mat = close_matrix(universe)
    n, n_assets = mat.shape
    space = {
        "lookback": (20, max(21, min(180, n - 10)), "int"),
        "top_n": (1, max(2, min(5, n_assets)), "int"),
        "reb_days": (5, 63, "int"),
    }
    cutoff = mat.index[int(n * train_frac) - 1] if train_frac < 1.0 else None

    def run(state: dict) -> Result:
        return run_rotation(universe, lookback=int(state["lookback"]),
                            top_n=int(state["top_n"]), reb_days=int(state["reb_days"]),
                            fee_bps=fee_bps, slippage_bps=slippage_bps)

    def score(state: dict) -> float:
        res = run(state)
        if cutoff is not None:
            r = res.returns[res.returns.index <= cutoff]
            if len(r) < 10:
                return float("-inf")
            res = Result(equity=(1 + r).cumprod(), returns=r, positions=r * 0)
        return objective_value(res, objective)

    def neighbor(state: dict, scale: float, rng: random.Random) -> dict:
        nxt = dict(state)
        key = rng.choice(list(space.keys()))
        lo, hi, kind = space[key]
        nxt[key] = _perturb_param(state[key], lo, hi, kind, scale, rng)
        return nxt

    init = {"lookback": 126, "top_n": 2, "reb_days": 21}
    init.update({k: v for k, v in (params or {}).items() if k in space})
    for k, (lo, hi, _) in space.items():
        init[k] = int(min(hi, max(lo, init[k])))

    log = anneal(init, neighbor, score, steps=steps, seed=seed, restarts=restarts,
                 on_step=on_step, should_stop=should_stop)
    best = {k: int(v) for k, v in log.best_state.items()}
    return best, run(log.best_state), log


# ---------------------------------------------------------------- Walk-forward

def walk_forward(universe: dict[str, pd.DataFrame], specs: list[dict],
                 train_days: int = 126, test_days: int = 21,
                 opt_weights: bool = True, opt_params: bool = True,
                 objective: str = "sharpe", fee_bps: float = 10.0,
                 slippage_bps: float = 0.0, vol_target: float = 0.0,
                  cash_rate: float = 0.0,
                 steps: int = 120, restarts: int = 1, seed: int = 42,
                 on_step: ProgressFn | None = None,
                 should_stop: StopFn | None = None) -> dict:
    """滾動視窗驗證：每段用「至今為止的資料、只看最後 train_days 評分」退火，
    把最佳解套用到下一段 test_days 的樣本外資料，拼接成 OOS 淨值曲線。
    每段以上一段的最佳解熱啟動。"""
    from .data import align_universe, slice_universe

    uni = align_universe(universe)
    n = len(next(iter(uni.values())))
    if n < train_days + test_days:
        raise ValueError(f"資料只有 {n} 天，不夠 train {train_days} + test {test_days}")

    seg_starts = list(range(train_days, n, test_days))
    total_steps = len(seg_starts) * steps * max(1, restarts)
    done_base = 0
    segments = []
    oos_parts = []
    cur_specs = specs

    for k, start in enumerate(seg_starts):
        if should_stop and should_stop():
            break
        end = min(start + test_days, n)
        train_uni = slice_universe(uni, 0, start)

        def seg_progress(done, total, best, _base=done_base):
            if on_step:
                on_step(_base + done, total_steps, best)

        best_specs, _, log = optimize_rows(
            train_uni, cur_specs, opt_weights=opt_weights, opt_params=opt_params,
            objective=objective, fee_bps=fee_bps, slippage_bps=slippage_bps,
            vol_target=vol_target, cash_rate=cash_rate, eval_slice=(max(0, start - train_days), start),
            steps=steps, restarts=restarts, seed=seed + k,
            on_step=seg_progress, should_stop=should_stop)
        done_base += steps * max(1, restarts)
        cur_specs = best_specs

        eval_uni = slice_universe(uni, 0, end)
        rows = [(s["ticker"], STRATEGIES[s["strategy"]](**s["params"]), s["weight"])
                for s in best_specs]
        res = run_rows(eval_uni, rows, fee_bps=fee_bps,
                       slippage_bps=slippage_bps, vol_target=vol_target, cash_rate=cash_rate)
        piece = res.returns.iloc[start:end]
        oos_parts.append(piece)
        seg_ret = float((1 + piece).prod() - 1)
        segments.append({
            "train_end": str(res.returns.index[start - 1].date()),
            "test_start": str(piece.index[0].date()) if len(piece) else "",
            "test_end": str(piece.index[-1].date()) if len(piece) else "",
            "train_score": round(log.best_score, 3),
            "oos_return": round(seg_ret, 4),
            "specs": [{"ticker": s["ticker"], "strategy": s["strategy"],
                       "weight": s["weight"]} for s in best_specs],
        })

    if not oos_parts:
        raise ValueError("Walk-forward 沒有產生任何樣本外區段")
    oos_ret = pd.concat(oos_parts)
    oos = Result(equity=(1 + oos_ret).cumprod(), returns=oos_ret, positions=oos_ret * 0,
                 trades=0)
    return {"result": oos, "segments": segments, "final_specs": cur_specs}


# ---------------------------------------------------------------- 參數穩健性掃描

def scan_params(universe: dict[str, pd.DataFrame], specs: list[dict], row_i: int,
                px: str, py: str, nx: int = 13, ny: int = 13,
                objective: str = "sharpe", fee_bps: float = 10.0,
                slippage_bps: float = 0.0, vol_target: float = 0.0,
                  cash_rate: float = 0.0,
                train_frac: float = 1.0) -> dict:
    """固定其他設定，掃描第 row_i 列策略的兩個參數網格，回傳目標函數值矩陣。
    好的參數解應該是一整片高原；孤立尖峰是過擬合警訊。"""
    cls = STRATEGIES[specs[row_i]["strategy"]]
    space = cls.full_space()
    if px not in space or py not in space:
        raise ValueError(f"參數必須是 {list(space)} 之一")

    def axis(p, k):
        lo, hi, kind = space[p]
        vals = np.linspace(lo, hi, k)
        if kind == "int":
            vals = sorted(set(int(round(v)) for v in vals))
        else:
            vals = [round(float(v), 4) for v in vals]
        return vals

    xs, ys = axis(px, nx), axis(py, ny)
    base_rows = []
    for s in specs:
        c = STRATEGIES[s["strategy"]]
        base_rows.append([s["ticker"], c, dict(c.full_defaults(), **(s.get("params") or {})),
                          float(s.get("weight", 1.0))])

    n_train = None
    z = []
    best = (float("-inf"), 0, 0)
    for yi, yv in enumerate(ys):
        row_vals = []
        for xi, xv in enumerate(xs):
            rows = []
            for i, (t, c, params, w) in enumerate(base_rows):
                p = dict(params)
                if i == row_i:
                    p[px], p[py] = xv, yv
                rows.append((t, c(**p), w))
            try:
                res = run_rows(universe, rows, fee_bps=fee_bps,
                               slippage_bps=slippage_bps, vol_target=vol_target, cash_rate=cash_rate)
                if n_train is None:
                    n_train = int(len(res.returns) * train_frac) if train_frac < 1.0 else -1
                if n_train > 0:
                    res = subresult(res, 0, n_train)
                v = objective_value(res, objective)
            except ValueError:
                v = float("nan")
            row_vals.append(round(v, 3) if not math.isnan(v) else None)
            if v == v and v > best[0]:
                best = (v, xi, yi)
        z.append(row_vals)
    return {"px": px, "py": py, "x": xs, "y": ys, "z": z,
            "best": {"x": xs[best[1]], "y": ys[best[2]], "value": round(best[0], 3)}}
