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
    p.add_argument("--restarts", type=int, default=1, help="多重啟次數（不同種子取最佳）")
    p.add_argument("--train-frac", type=float, default=1.0,
                   help="<1 時只在前段（訓練期）評分，其餘留作驗證")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fee-bps", type=float, default=10.0)
    p.add_argument("--slippage-bps", type=float, default=0.0)

    p = sub.add_parser("walkforward", help="滾動視窗：逐段退火 → 下一段樣本外驗證")
    p.add_argument("-s", "--strategy", choices=STRATEGIES.keys(), required=True)
    p.add_argument("-t", "--tickers", nargs="+", required=True)
    p.add_argument("--train-days", type=int, default=126)
    p.add_argument("--test-days", type=int, default=21)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--objective", default="sharpe",
                   choices=["sharpe", "total_return", "calmar_mix"])
    p.add_argument("--fee-bps", type=float, default=10.0)

    p = sub.add_parser("dip", help="逢跌加碼研究：多檔 × 多種觸發幅度，對照買進持有")
    p.add_argument("-t", "--tickers", nargs="+", required=True)
    p.add_argument("--triggers", nargs="+", type=float, default=[0.02, 0.03, 0.05, 0.08, 0.10],
                   help="回撤觸發幅度，例如 0.03 = 3%%")
    p.add_argument("--buy-fracs", nargs="+", type=float, default=[0.05],
                   help="每次投入初始資金的比例，例如 0.05 = 5%%")
    p.add_argument("--mode", type=int, default=0, choices=[0, 1, 2],
                   help="0 每跌一階買一次 / 1 從上次買價再跌一階 / 2 回撤中每 cooldown 天買")
    p.add_argument("--cooldown", type=int, default=5)
    p.add_argument("--init-frac", type=float, default=0.0, help="第一天先投入的比例")
    p.add_argument("--cash-rate", type=float, default=0.0, help="閒置現金年利率，例如 0.04")
    p.add_argument("--fee-bps", type=float, default=10.0)

    p = sub.add_parser("rotation", help="動量輪動：每期買過去表現最強的 N 檔")
    p.add_argument("-t", "--tickers", nargs="*", default=None, help="預設用 data 目錄全部")
    p.add_argument("--lookback", type=int, default=126)
    p.add_argument("--top-n", type=int, default=2)
    p.add_argument("--reb-days", type=int, default=21)
    p.add_argument("--fee-bps", type=float, default=10.0)
    p.add_argument("--anneal", action="store_true", help="先退火找最佳輪動參數")

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
            _print({k: v for k, v in res.summary().items() if k != "trades_log"})

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
        from .backtest import subresult
        df = data.load_csv(f"{args.data_dir}/{args.ticker}.csv")
        print(f"對 {args.ticker} 的 {args.strategy} 做模擬退火"
              f"（{args.steps} 步 × {args.restarts} 重啟，目標 {args.objective}）...")
        best, res, log = annealing.optimize_params(
            df, args.strategy, objective=args.objective, fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps, train_frac=args.train_frac,
            steps=args.steps, restarts=args.restarts, seed=args.seed, verbose=True)
        print(f"\n最佳參數: {best.params}")
        summary = {k: v for k, v in res.summary().items() if k != "trades_log"}
        _print(summary)
        if args.train_frac < 1.0:
            nt = int(len(res.returns) * args.train_frac)
            tr, va = subresult(res, 0, nt), subresult(res, nt, None)
            print(f"訓練期 Sharpe {tr.sharpe:.2f} / 報酬 {tr.total_return:.2%}  |  "
                  f"驗證期 Sharpe {va.sharpe:.2f} / 報酬 {va.total_return:.2%}")

    elif args.cmd == "walkforward":
        uni = data.load_universe(args.data_dir, tickers=args.tickers)
        specs = [{"ticker": t, "strategy": args.strategy, "params": {},
                  "weight": 1.0 / len(uni)} for t in uni]
        print(f"Walk-forward：訓練 {args.train_days} 天 / 驗證 {args.test_days} 天 ...")
        wf = annealing.walk_forward(uni, specs, train_days=args.train_days,
                                    test_days=args.test_days, steps=args.steps,
                                    objective=args.objective, fee_bps=args.fee_bps)
        res = wf["result"]
        print(f"\n樣本外：報酬 {res.total_return:.2%}  Sharpe {res.sharpe:.2f}  "
              f"最大回撤 {res.max_drawdown:.2%}")
        for g in wf["segments"]:
            print(f"  {g['test_start']} ~ {g['test_end']}  訓練分數 {g['train_score']:+.2f}  "
                  f"樣本外 {g['oos_return']:+.2%}")

    elif args.cmd == "dip":
        for t in args.tickers:
            df = data.load_csv(f"{args.data_dir}/{t}.csv")
            dd = (df["close"] / df["close"].cummax() - 1).min()
            bh = run_signals(df, make("buyhold"), fee_bps=args.fee_bps)
            print(f"\n=== {t}  {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}  "
                  f"（期間最大回撤 {dd:.1%}）===")
            print(f"{'設定':<22}{'報酬':>9}{'最大回撤':>9}{'Sharpe':>8}"
                  f"{'買入次數':>8}{'平均持股':>8}{'期末持股':>8}")
            print(f"{'買進持有（100%）':<20}{bh.total_return:>9.2%}{bh.max_drawdown:>9.2%}"
                  f"{bh.sharpe:>8.2f}{1:>8}{1:>8.0%}{1:>8.0%}")
            for frac in args.buy_fracs:
                for trig in args.triggers:
                    strat = make("dip", dd_trigger=trig, buy_frac=frac, mode=args.mode,
                                 cooldown=args.cooldown, init_frac=args.init_frac)
                    res = run_signals(df, strat, fee_bps=args.fee_bps, cash_rate=args.cash_rate)
                    label = f"跌{trig:.0%}買{frac:.0%}"
                    print(f"{label:<24}{res.total_return:>9.2%}{res.max_drawdown:>9.2%}"
                          f"{res.sharpe:>8.2f}{res.trades:>8}"
                          f"{res.positions.mean():>8.0%}{res.positions.iloc[-1]:>8.0%}")

    elif args.cmd == "rotation":
        from .backtest import run_rotation
        uni = data.align_universe(data.load_universe(args.data_dir, tickers=args.tickers))
        if args.anneal:
            best, res, log = annealing.optimize_rotation(
                uni, params={"lookback": args.lookback, "top_n": args.top_n,
                             "reb_days": args.reb_days}, fee_bps=args.fee_bps)
            print(f"最佳輪動參數: {best}")
        else:
            res = run_rotation(uni, lookback=args.lookback, top_n=args.top_n,
                               reb_days=args.reb_days, fee_bps=args.fee_bps)
        print(f"報酬 {res.total_return:.2%}  Sharpe {res.sharpe:.2f}  "
              f"最大回撤 {res.max_drawdown:.2%}")
        for h in res.meta.get("holdings", []):
            print(f"  {h['date']} 持有 {h['hold']}")

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
