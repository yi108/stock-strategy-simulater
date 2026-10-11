"""本機網頁介面：自由排列「股票 × 策略 × 參數 × 權重」並回測 / 模擬退火。

啟動：
    python -m simulator.web [--port 8000] [--data-dir data]
然後開瀏覽器到 http://127.0.0.1:8000

長時間工作（退火 / walk-forward）以背景執行緒執行，
前端輪詢 /api/job?id=... 取得進度，POST /api/cancel 可中止（回傳目前最佳解）。
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pandas as pd

from . import annealing, data
from .backtest import run_rotation, run_rows, subresult
from .strategies import STRATEGIES, make

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
DATA_DIR = "data"


# ---------------------------------------------------------------- 背景工作

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def start_job(fn, body: dict) -> dict:
    jid = uuid.uuid4().hex[:12]
    cancel = threading.Event()
    with JOBS_LOCK:
        JOBS[jid] = {"status": "running", "progress": {"done": 0, "total": 1, "best": None},
                     "live": [], "result": None, "error": None, "cancel": cancel}
        if len(JOBS) > 40:   # 清掉最舊的已完成工作
            for k in list(JOBS):
                if k != jid and JOBS[k]["status"] != "running":
                    del JOBS[k]
                    break

    def on_step(done, total, best):
        j = JOBS.get(jid)
        if j:
            j["progress"] = {"done": done, "total": total, "best": round(best, 4)}
            j["live"].append([done, round(best, 4)])
            if len(j["live"]) > 400:
                j["live"] = j["live"][::2]

    def run():
        try:
            JOBS[jid]["result"] = fn(body, on_step, cancel.is_set)
            JOBS[jid]["status"] = "done"
        except Exception as e:
            traceback.print_exc()
            JOBS[jid]["error"] = f"{type(e).__name__}: {e}"
            JOBS[jid]["status"] = "error"

    threading.Thread(target=run, daemon=True).start()
    return {"job": jid}


# ---------------------------------------------------------------- API 實作

def api_strategies() -> dict:
    out = {}
    for name, cls in STRATEGIES.items():
        params = {}
        defaults = cls.full_defaults()
        for k, (lo, hi, kind) in cls.full_space().items():
            params[k] = {"lo": lo, "hi": hi, "kind": kind, "default": defaults[k]}
        out[name] = {"doc": (cls.__doc__ or "").strip().split("\n")[0], "params": params}
    return out


def api_tickers() -> list[dict]:
    out = []
    if os.path.isdir(DATA_DIR):
        for f in sorted(os.listdir(DATA_DIR)):
            if not f.endswith(".csv"):
                continue
            try:
                df = data.load_csv(os.path.join(DATA_DIR, f))
                out.append({"ticker": f[:-4], "bars": len(df),
                            "start": df["date"].iloc[0], "end": df["date"].iloc[-1]})
            except Exception:
                continue
    return out


def _load_universe(tickers) -> dict[str, pd.DataFrame]:
    uni = {}
    for t in sorted(set(tickers)):
        path = os.path.join(DATA_DIR, f"{t}.csv")
        if not os.path.exists(path):
            raise ValueError(f"找不到 {t} 的資料（{path}）")
        uni[t] = data.load_csv(path)
    return uni


def _common(body: dict) -> dict:
    return {"fee_bps": float(body.get("fee_bps", 10.0)),
            "slippage_bps": float(body.get("slippage_bps", 0.0)),
            "vol_target": float(body.get("vol_target", 0.0))}


def _benchmark(universe, fee_bps: float):
    rows = [(t, make("buyhold"), 1.0) for t in universe]
    return run_rows(universe, rows, fee_bps=fee_bps)


def _series(s) -> list[float]:
    return [round(float(v), 4) for v in s]


def _plain(res) -> dict:
    return {k: v for k, v in res.summary().items()
            if k in ("total_return", "cagr", "sharpe", "max_drawdown", "trades")}


def _result_payload(res, universe, fee_bps: float, train_frac: float = 1.0) -> dict:
    bench = _benchmark(universe, fee_bps)
    idx = res.equity.index
    bench_eq = bench.equity.reindex(idx).ffill()
    rows = []
    for r in res.meta.get("rows", []):
        rr = r["result"]
        rows.append({
            "label": f'{r["ticker"]} · {r["strategy"].NAME}',
            "weight": r["weight"],
            "equity": _series(rr.equity.reindex(idx).ffill().fillna(1.0)),
            "summary": _plain(rr),
            "trades": rr.meta.get("trades_log", [])[:60],
        })
    payload = {
        "summary": _plain(res),
        "dates": [d.strftime("%Y-%m-%d") for d in idx],
        "portfolio": _series(res.equity),
        "drawdown": _series(res.drawdown),
        "benchmark": _series(bench_eq),
        "benchmark_summary": _plain(bench),
        "rows": rows,
    }
    if train_frac < 1.0:
        n = len(res.returns)
        nt = int(n * train_frac)
        if 10 < nt < n - 5:
            payload["split"] = {
                "index": nt,
                "date": idx[nt].strftime("%Y-%m-%d"),
                "train": _plain(subresult(res, 0, nt)),
                "valid": _plain(subresult(res, nt, None)),
            }
    return payload


def _specs_to_rows(specs):
    return [(s["ticker"], make(s["strategy"], **(s.get("params") or {})),
             float(s.get("weight", 1.0))) for s in specs]


def api_backtest(body: dict) -> dict:
    specs = body["rows"]
    c = _common(body)
    train_frac = float(body.get("train_frac", 1.0))
    uni = _load_universe(s["ticker"] for s in specs)
    res = run_rows(uni, _specs_to_rows(specs), **c)
    return _result_payload(res, uni, c["fee_bps"], train_frac)


def job_anneal(body: dict, on_step, should_stop) -> dict:
    specs = body["rows"]
    c = _common(body)
    train_frac = float(body.get("train_frac", 1.0))
    uni = _load_universe(s["ticker"] for s in specs)
    best_specs, res, log = annealing.optimize_rows(
        uni, specs,
        opt_weights=bool(body.get("opt_weights", True)),
        opt_params=bool(body.get("opt_params", True)),
        objective=body.get("objective", "sharpe"),
        train_frac=train_frac,
        steps=int(body.get("steps", 500)),
        restarts=int(body.get("restarts", 1)),
        seed=int(body.get("seed", 42)),
        on_step=on_step, should_stop=should_stop, **c)
    payload = _result_payload(res, uni, c["fee_bps"], train_frac)
    payload["best_rows"] = best_specs
    payload["best_score"] = round(log.best_score, 4)
    payload["stopped"] = log.stopped
    payload["history"] = [[s, round(b, 4)] for s, _, b in log.history]
    return payload


def job_walkforward(body: dict, on_step, should_stop) -> dict:
    specs = body["rows"]
    c = _common(body)
    uni = _load_universe(s["ticker"] for s in specs)
    wf = annealing.walk_forward(
        uni, specs,
        train_days=int(body.get("train_days", 126)),
        test_days=int(body.get("test_days", 21)),
        opt_weights=bool(body.get("opt_weights", True)),
        opt_params=bool(body.get("opt_params", True)),
        objective=body.get("objective", "sharpe"),
        steps=int(body.get("steps", 120)),
        restarts=int(body.get("restarts", 1)),
        seed=int(body.get("seed", 42)),
        on_step=on_step, should_stop=should_stop, **c)
    res = wf["result"]
    idx = res.equity.index
    aligned = data.align_universe(uni)
    bench = _benchmark(aligned, c["fee_bps"])
    bench_ret = bench.returns.reindex(idx).fillna(0.0)
    bench_eq = (1 + bench_ret).cumprod()
    return {
        "summary": _plain(res),
        "benchmark_summary": _plain(
            subresult(bench, len(bench.returns) - len(idx), None)),
        "dates": [d.strftime("%Y-%m-%d") for d in idx],
        "portfolio": _series(res.equity),
        "drawdown": _series(res.drawdown),
        "benchmark": _series(bench_eq),
        "segments": wf["segments"],
        "final_specs": wf["final_specs"],
    }


def api_scan(body: dict) -> dict:
    specs = body["rows"]
    c = _common(body)
    uni = _load_universe(s["ticker"] for s in specs)
    return annealing.scan_params(
        uni, specs, int(body["row_i"]), body["px"], body["py"],
        nx=int(body.get("nx", 13)), ny=int(body.get("ny", 13)),
        objective=body.get("objective", "sharpe"),
        train_frac=float(body.get("train_frac", 1.0)), **c)


def _rotation_payload(res, uni, fee_bps) -> dict:
    idx = res.equity.index
    bench = _benchmark(uni, fee_bps)
    bench_ret = bench.returns.reindex(idx).fillna(0.0)
    bench_eq = (1 + bench_ret).cumprod()
    return {
        "summary": _plain(res),
        "benchmark_summary": _plain(
            subresult(bench, len(bench.returns) - len(idx), None)),
        "dates": [d.strftime("%Y-%m-%d") for d in idx],
        "portfolio": _series(res.equity),
        "drawdown": _series(res.drawdown),
        "benchmark": _series(bench_eq),
        "holdings": res.meta.get("holdings", []),
        "params": res.meta.get("rotation", {}),
    }


def api_rotation(body: dict) -> dict:
    tickers = body["tickers"]
    if len(tickers) < 2:
        raise ValueError("輪動策略至少需要 2 檔標的")
    c = _common(body)
    uni = data.align_universe(_load_universe(tickers))
    res = run_rotation(uni, lookback=int(body.get("lookback", 126)),
                       top_n=int(body.get("top_n", 2)),
                       reb_days=int(body.get("reb_days", 21)),
                       fee_bps=c["fee_bps"], slippage_bps=c["slippage_bps"])
    return _rotation_payload(res, uni, c["fee_bps"])


def job_rotation_anneal(body: dict, on_step, should_stop) -> dict:
    tickers = body["tickers"]
    if len(tickers) < 2:
        raise ValueError("輪動策略至少需要 2 檔標的")
    c = _common(body)
    uni = data.align_universe(_load_universe(tickers))
    best, res, log = annealing.optimize_rotation(
        uni, params={"lookback": int(body.get("lookback", 126)),
                     "top_n": int(body.get("top_n", 2)),
                     "reb_days": int(body.get("reb_days", 21))},
        objective=body.get("objective", "sharpe"),
        train_frac=float(body.get("train_frac", 1.0)),
        steps=int(body.get("steps", 300)),
        restarts=int(body.get("restarts", 1)),
        seed=int(body.get("seed", 42)),
        fee_bps=c["fee_bps"], slippage_bps=c["slippage_bps"],
        on_step=on_step, should_stop=should_stop)
    payload = _rotation_payload(res, uni, c["fee_bps"])
    payload["best_params"] = best
    payload["best_score"] = round(log.best_score, 4)
    payload["stopped"] = log.stopped
    payload["history"] = [[s, round(b, 4)] for s, _, b in log.history]
    return payload


def api_sample(body: dict) -> dict:
    uni = data.generate_sample_universe(DATA_DIR, tickers=body.get("tickers") or None,
                                        days=int(body.get("days", 252)))
    return {"created": list(uni)}


def api_fetch(body: dict) -> dict:
    out = []
    for t in body["tickers"]:
        df = data.fetch(t, period=body.get("period", "1y"))
        data.save_csv(df, os.path.join(DATA_DIR, f"{t}.csv"))
        out.append({"ticker": t, "bars": len(df)})
    return {"fetched": out}


# ---------------------------------------------------------------- HTTP 伺服器

SYNC_ROUTES = {"/api/backtest": api_backtest, "/api/scan": api_scan,
               "/api/rotation": api_rotation, "/api/sample": api_sample,
               "/api/fetch": api_fetch}
JOB_ROUTES = {"/api/anneal": job_anneal, "/api/walkforward": job_walkforward,
              "/api/rotation_anneal": job_rotation_anneal}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        path, _, query = self.path.partition("?")
        if path in ("/", "/index.html"):
            with open(os.path.join(STATIC_DIR, "index.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif path == "/api/strategies":
            self._json(api_strategies())
        elif path == "/api/tickers":
            self._json(api_tickers())
        elif path == "/api/job":
            jid = dict(p.split("=", 1) for p in query.split("&") if "=" in p).get("id", "")
            j = JOBS.get(jid)
            if not j:
                self._json({"error": "沒有這個工作"}, 404)
            else:
                self._json({"status": j["status"], "progress": j["progress"],
                            "live": j["live"][-200:], "error": j["error"],
                            "result": j["result"] if j["status"] == "done" else None})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json({"error": "JSON 格式錯誤"}, 400)
            return
        path = self.path.partition("?")[0]
        try:
            if path in SYNC_ROUTES:
                self._json(SYNC_ROUTES[path](body))
            elif path in JOB_ROUTES:
                self._json(start_job(JOB_ROUTES[path], body))
            elif path == "/api/cancel":
                j = JOBS.get(body.get("job", ""))
                if j:
                    j["cancel"].set()
                self._json({"ok": True})
            else:
                self._json({"error": "not found"}, 404)
        except (ValueError, KeyError) as e:
            self._json({"error": str(e)}, 400)
        except Exception as e:
            traceback.print_exc()
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def log_message(self, fmt, *args):  # 安靜一點
        pass


def main(argv: list[str] | None = None) -> None:
    global DATA_DIR
    ap = argparse.ArgumentParser(description="股票策略模擬器網頁介面")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args(argv)
    DATA_DIR = args.data_dir
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"策略模擬器介面: http://{args.host}:{args.port}  (資料目錄: {DATA_DIR})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
