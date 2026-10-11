"""本機網頁介面：自由排列「股票 × 策略 × 參數 × 權重」並回測 / 模擬退火。

啟動：
    python -m simulator.web [--port 8000] [--data-dir data]
然後開瀏覽器到 http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pandas as pd

from . import annealing, data
from .backtest import run_rows
from .strategies import STRATEGIES, make

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
DATA_DIR = "data"


# ---------------------------------------------------------------- API 實作

def api_strategies() -> dict:
    out = {}
    for name, cls in STRATEGIES.items():
        params = {}
        defaults = cls.default_params()
        for k, (lo, hi, kind) in cls.PARAM_SPACE.items():
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


def _load_universe_for(specs: list[dict]) -> dict[str, pd.DataFrame]:
    tickers = sorted({s["ticker"] for s in specs})
    uni = {}
    for t in tickers:
        path = os.path.join(DATA_DIR, f"{t}.csv")
        if not os.path.exists(path):
            raise ValueError(f"找不到 {t} 的資料（{path}）")
        uni[t] = data.load_csv(path)
    return uni


def _benchmark(universe: dict[str, pd.DataFrame], fee_bps: float):
    """等權買進持有基準。"""
    rows = [(t, make("buyhold"), 1.0) for t in universe]
    return run_rows(universe, rows, fee_bps=fee_bps)


def _series(s) -> list[float]:
    return [round(float(v), 4) for v in s]


def _clean_summary(res) -> dict:
    return {k: v for k, v in res.summary().items() if k != "rows"}


def _result_payload(res, universe, fee_bps) -> dict:
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
            "summary": rr.summary(),
        })
    return {
        "summary": _clean_summary(res),
        "dates": [d.strftime("%Y-%m-%d") for d in idx],
        "portfolio": _series(res.equity),
        "benchmark": _series(bench_eq),
        "benchmark_summary": _clean_summary(bench),
        "rows": rows,
    }


def _specs_to_rows(specs: list[dict]):
    rows = []
    for s in specs:
        strat = make(s["strategy"], **(s.get("params") or {}))
        rows.append((s["ticker"], strat, float(s.get("weight", 1.0))))
    return rows


def api_backtest(body: dict) -> dict:
    specs = body["rows"]
    fee = float(body.get("fee_bps", 10.0))
    uni = _load_universe_for(specs)
    res = run_rows(uni, _specs_to_rows(specs), fee_bps=fee)
    return _result_payload(res, uni, fee)


def api_anneal(body: dict) -> dict:
    specs = body["rows"]
    fee = float(body.get("fee_bps", 10.0))
    uni = _load_universe_for(specs)
    best_specs, res, log = annealing.optimize_rows(
        uni, specs,
        opt_weights=bool(body.get("opt_weights", True)),
        opt_params=bool(body.get("opt_params", True)),
        objective=body.get("objective", "sharpe"),
        fee_bps=fee,
        steps=int(body.get("steps", 600)),
        seed=int(body.get("seed", 42)),
    )
    payload = _result_payload(res, uni, fee)
    payload["best_rows"] = best_specs
    payload["best_score"] = round(log.best_score, 4)
    payload["history"] = [[s, round(t, 5), round(b, 4)] for s, t, b in log.history]
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
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            with open(os.path.join(STATIC_DIR, "index.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif path == "/api/strategies":
            self._json(api_strategies())
        elif path == "/api/tickers":
            self._json(api_tickers())
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json({"error": "JSON 格式錯誤"}, 400)
            return
        routes = {"/api/backtest": api_backtest, "/api/anneal": api_anneal,
                  "/api/sample": api_sample, "/api/fetch": api_fetch}
        fn = routes.get(self.path.split("?")[0])
        if fn is None:
            self._json({"error": "not found"}, 404)
            return
        try:
            self._json(fn(body))
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
