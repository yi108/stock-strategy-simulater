"""命令列介面。

  python -m simulator.cli sample                       產生離線示範資料
  python -m simulator.cli fetch -t AAPL MSFT           抓過去一年真實報價（需網路）
  python -m simulator.cli record -t AAPL MSFT          每日記錄最新報價（append）
  python -m simulator.cli backtest -s macd -t DEMO_A   單一策略回測
  python -m simulator.cli compare -t DEMO_A            所有策略比一比
  python -m simulator.cli optimize -s macd -t DEMO_A   退火找最佳參數
  python -m simulator.cli optimize-portfolio -s macd_boll   退火找組合權重+參數
"""

from __future__ import annotations

import argparse
import json

from . import annealing, data
from .backtest import run_portfolio, run_signals
from .strategies import STRATEGIES, make


def _parse_params(pairs: list[str]) -> dict:
    out = {}
    for p in pairs or []:
        k, v = p.split("=", 1)
        try:
            out[k] = int(v)
        except ValueError:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def _print(obj):
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="simulator", description="股票策略模擬器")
    ap.add_argument("--data-dir", default="data", help="CSV 資料目錄（預設 data/）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("sample", help="產生合成示範資料")
    p.add_argument("--days", type=int, default=252)
    p.add_argument("-t", "--tickers", nargs="*", default=None)

    p = sub.add_parser("fetch", help="抓過去一段期間的真實報價（需網路）")
    p.add_argument("-t", "--tickers", nargs="+", required=True)
    p.add_argument("--period", default="1y", help="例如 1y、2y、6mo")

    p = sub.add_parser("record", help="抓最新報價並 append 到 CSV（每日執行）")
    p.add_argument("-t", "--tickers", nargs="+", required=True)

    p = sub.add_parser("backtest", help="回測單一策略")
    p.add_argument("-s", "--strategy", choices=STRATEGIES.keys(), required=True)
    p.add_argument("-t", "--tickers", nargs="+", required=True)
    p.add_argument("-p", "--param", nargs="*", help="覆寫參數，如 fast=8 slow=30")
    p.add_argument("--fee-bps", type=float, default=10.0)

    p = sub.add_parser("compare", help="同一標的上比較所有策略")
    p.add_argument("-t", "--tickers", nargs="+", required=True)
    p.add_argument("--fee-bps", type=float, default=10.0)

    p = sub.add_parser("optimize", help="模擬退火：最佳化策略參數")
    p.add_argument("-s", "--strategy", choices=STRATEGIES.keys(), required=True)
    p.add_argument("-t", "--ticker", required=True)
    p.add_argument("--objective", default="sharpe",
                   choices=["sharpe", "total_return", "calmar_mix"])
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fee-bps", type=float, default=10.0)

    p = sub.add_parser("optimize-portfolio", help="模擬退火：組合權重 + 各標的參數")
    p.add_argument("-s", "--strategy", choices=STRATEGIES.keys(), required=True)
    p.add_argument("-t", "--tickers", nargs="*", default=None, help="預設用 data 目錄全部")
    p.add_argument("--objective", default="sharpe",
                   choices=["sharpe", "total_return", "calmar_mix"])
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fee-bps", type=float, default=10.0)

    args = ap.parse_args(argv)

    if args.cmd == "sample":
        uni = data.generate_sample_universe(args.data_dir, tickers=args.tickers, days=args.days)
        print(f"已產生 {len(uni)} 檔示範資料到 {args.data_dir}/: {', '.join(uni)}")

    elif args.cmd == "fetch":
        for t in args.tickers:
            df = data.fetch(t, period=args.period)
            data.save_csv(df, f"{args.data_dir}/{t}.csv")
            print(f"{t}: {len(df)} 根日K ({df['date'].iloc[0]} ~ {df['date'].iloc[-1]})")

    elif args.cmd == "record":
        for t in args.tickers:
            df = data.record(t, data_dir=args.data_dir)
            print(f"{t}: 累計 {len(df)} 根日K，最新 {df['date'].iloc[-1]}")

    elif args.cmd == "backtest":
        strat = make(args.strategy, **_parse_params(args.param))
        for t in args.tickers:
            df = data.load_csv(f"{args.data_dir}/{t}.csv")
            res = run_signals(df, strat, fee_bps=args.fee_bps)
            print(f"\n=== {t} / {strat!r} ===")
            _print(res.summary())

    elif args.cmd == "compare":
        for t in args.tickers:
            df = data.load_csv(f"{args.data_dir}/{t}.csv")
            print(f"\n=== {t} ===")
            rows = []
            for name in STRATEGIES:
                res = run_signals(df, make(name), fee_bps=args.fee_bps)
                s = res.summary()
                rows.append((name, s["total_return"], s["sharpe"], s["max_drawdown"], s["trades"]))
            print(f"{'策略':<12}{'總報酬':>10}{'Sharpe':>10}{'最大回撤':>10}{'交易次數':>8}")
            for name, tr, sh, dd, n in rows:
                print(f"{name:<12}{tr:>10.2%}{sh:>10.2f}{dd:>10.2%}{n:>8}")

    elif args.cmd == "optimize":
        df = data.load_csv(f"{args.data_dir}/{args.ticker}.csv")
        print(f"對 {args.ticker} 的 {args.strategy} 做模擬退火（{args.steps} 步，目標 {args.objective}）...")
        best, res, log = annealing.optimize_params(
            df, args.strategy, objective=args.objective, fee_bps=args.fee_bps,
            steps=args.steps, seed=args.seed, verbose=True)
        print(f"\n最佳參數: {best.params}")
        _print(res.summary())

    elif args.cmd == "optimize-portfolio":
        uni = data.load_universe(args.data_dir, tickers=args.tickers)
        print(f"對 {len(uni)} 檔標的做組合退火（策略 {args.strategy}，{args.steps} 步，目標 {args.objective}）...")
        best, res, log = annealing.optimize_portfolio(
            uni, args.strategy, objective=args.objective, fee_bps=args.fee_bps,
            steps=args.steps, seed=args.seed, verbose=True)
        print("\n最佳組合權重:", json.dumps(best["weights"], ensure_ascii=False))
        _print(res.summary())


if __name__ == "__main__":
    main()
