"""命令列介面。

  python -m simulator.cli sample                       產生離線示範資料
  python -m simulator.cli fetch -t AAPL MSFT           抓過去一年真實報價（需網路）
  python -m simulator.cli record -t AAPL MSFT          每日記錄最新報價（append）
  python -m simulator.cli backtest -s macd -t DEMO_A   單一策略回測
  python -m simulator.cli compare -t DEMO_A            所有策略比一比
  python -m simulator.cli optimize -s macd -t DEMO_A   退火找最佳參數
  python -m simulator.cli optimize-portfolio -s macd_boll   退火找組合權重+參數
  python -m simulator.cli dip -t QQQ QLD                逢跌加碼 vs 買進持有 vs 固定配置
  python -m simulator.cli rolling -t QQQ QLD            逢跌加碼多起點滾動測試（10 年）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata

import numpy as np
import pandas as pd

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


def _width(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)


def _table(header: list[str], rows: list[list[str]]) -> None:
    """依「顯示寬度」對齊的表格（中文字算兩格）；第一欄靠左、其餘靠右。"""
    cols = [header] + rows
    widths = [max(_width(r[i]) for r in cols) for i in range(len(header))]
    for r in cols:
        cells = []
        for i, cell in enumerate(r):
            pad = " " * (widths[i] - _width(cell))
            cells.append(cell + pad if i == 0 else pad + cell)
        print("  ".join(cells))


def _pct(v, digits: int = 2) -> str:
    return "—" if v is None or v != v else f"{v:.{digits}%}"


def main(argv: list[str] | None = None) -> None:
    try:
        _main(argv)
    except data.DataError as e:
        print(f"錯誤：{e}", file=sys.stderr)
        sys.exit(1)


def _main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="simulator", description="股票策略模擬器")
    ap.add_argument("--data-dir", default="data", help="CSV 資料目錄（預設 data/）")
    ap.add_argument("--fetch-period", default="1y",
                    help="資料檔不存在時自動下載的期間（預設 1y）")
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
    p.add_argument("--exit-mode", type=int, default=0, choices=[0, 1, 2],
                   help="0 只買不賣 / 1 回到前高全部賣出 / 2 部位獲利達 --tp-pct 全部賣出")
    p.add_argument("--tp-pct", type=float, default=0.10)
    p.add_argument("--cash-rate", default="0",
                   help="閒置現金年利率，例如 0.04；auto = 用 13 週國庫券(^IRX)每日實際利率")
    p.add_argument("--fee-bps", type=float, default=10.0)

    p = sub.add_parser("rolling", help="逢跌加碼多起點滾動測試：每月一個起點各跑一年，對照買進持有")
    p.add_argument("-t", "--tickers", nargs="+", required=True)
    p.add_argument("--years", type=int, default=10, help="使用最近幾年的歷史（預設 10）")
    p.add_argument("--window-days", type=int, default=252, help="每段長度（交易日，預設 252 ≈ 一年）")
    p.add_argument("--step-days", type=int, default=21, help="起點間隔（交易日，預設 21 ≈ 一個月）")
    p.add_argument("--triggers", nargs="+", type=float, default=[0.03, 0.05, 0.10])
    p.add_argument("--buy-fracs", nargs="+", type=float, default=[0.05, 0.10])
    p.add_argument("--mode", type=int, default=0, choices=[0, 1, 2])
    p.add_argument("--cooldown", type=int, default=5)
    p.add_argument("--init-frac", type=float, default=0.0)
    p.add_argument("--exit-mode", type=int, default=0, choices=[0, 1, 2],
                   help="0 只買不賣 / 1 回到前高全部賣出 / 2 部位獲利達 --tp-pct 全部賣出")
    p.add_argument("--tp-pct", type=float, default=0.10)
    p.add_argument("--cash-rate", default="auto",
                   help="auto（預設）= ^IRX 國庫券每日實際利率；或固定值如 0.04")
    p.add_argument("--fee-bps", type=float, default=10.0)
    p.add_argument("--out-dir", default="results", help="每段明細 CSV 的輸出目錄")

    p = sub.add_parser("synth", help="用標的每日報酬合成槓桿 ETF，補足上市前的歷史")
    p.add_argument("-t", "--base", default="QQQ", help="標的（預設 QQQ）")
    p.add_argument("--leverage", type=float, default=2.0)
    p.add_argument("--fee", type=float, default=0.0095, help="年管理費（預設 0.95%%，QLD 的費用率）")
    p.add_argument("--name", default="QLD2X", help="輸出代號，存成 data/<name>.csv")
    p.add_argument("--years", type=int, default=26)
    p.add_argument("--check", default="QLD", help="與真實 ETF 比對準確度（空字串可略過）")

    p = sub.add_parser("best", help="策略實驗室：多種策略同場比較，依可承受回撤挑出最佳")
    p.add_argument("-t", "--tickers", nargs="+", default=["QQQ", "QLD"],
                   help="第一檔為主要標的（預設 QQQ QLD）；多檔時只用共同日期")
    p.add_argument("--years", type=int, default=20)
    p.add_argument("--window-days", type=int, default=252)
    p.add_argument("--step-days", type=int, default=21)
    p.add_argument("--sma-days", type=int, default=200, help="均線濾網天數（預設 200）")
    p.add_argument("--short-fee", type=float, default=0.0095,
                   help="做空年費用（以反向 ETF 計，預設 0.95%%）；空單另賺國庫券利率")
    p.add_argument("--cash-rate", default="auto")
    p.add_argument("--fee-bps", type=float, default=10.0)
    p.add_argument("--out-dir", default="results")

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

    def load(ticker: str):
        return data.load_or_fetch(ticker, args.data_dir, period=args.fetch_period)

    def cash_for(df, years: int = 10):
        """--cash-rate 轉成固定值或與 df 對齊的每日利率序列。"""
        if str(args.cash_rate).lower() == "auto":
            return data.load_cash_rates(df["date"], args.data_dir, years=years)
        try:
            return float(args.cash_rate)
        except ValueError:
            raise data.DataError(f"--cash-rate 要是數字（如 0.04）或 auto，收到 {args.cash_rate!r}")

    def load_uni(tickers):
        if tickers:
            return {t: load(t) for t in tickers}
        uni = data.load_universe(args.data_dir) if os.path.isdir(args.data_dir) else {}
        if not uni:
            raise data.DataError(f"{args.data_dir}/ 裡沒有任何 CSV，請用 -t 指定代號或先執行 fetch")
        return uni

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
            df = load(t)
            res = run_signals(df, strat, fee_bps=args.fee_bps)
            print(f"\n=== {t} / {strat!r} ===")
            _print({k: v for k, v in res.summary().items() if k != "trades_log"})

    elif args.cmd == "compare":
        for t in args.tickers:
            df = load(t)
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
        df = load(args.ticker)
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
        uni = load_uni(args.tickers)
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
        from .backtest import run_constant_mix
        from .research import dip_configs
        for t in args.tickers:
            df = load(t)
            span = (pd.Timestamp(df["date"].iloc[-1]) - pd.Timestamp(df["date"].iloc[0])).days
            cash = cash_for(df, years=max(1, -(-span // 365)))
            dd = (df["close"] / df["close"].cummax() - 1).min()
            bh = run_signals(df, make("buyhold"), fee_bps=args.fee_bps)
            print(f"\n=== {t}  {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}  "
                  f"（期間最大回撤 {dd:.1%}）===")
            rows = [["買進持有（100%）", _pct(bh.total_return), _pct(bh.max_drawdown),
                     f"{bh.sharpe:.2f}", "1", "100%", "100%", "—", "—"]]
            for c in dip_configs(args.triggers, args.buy_fracs, args.mode,
                                 args.cooldown, args.init_frac, args.exit_mode, args.tp_pct):
                params = {k: v for k, v in c.items() if k != "label"}
                res = run_signals(df, make("dip", **params), fee_bps=args.fee_bps,
                                  cash_rate=cash)
                expo = float(res.positions.mean())
                mix = run_constant_mix(df, expo, cash)
                rows.append([c["label"], _pct(res.total_return), _pct(res.max_drawdown),
                             f"{res.sharpe:.2f}", str(res.trades), _pct(expo, 0),
                             _pct(float(res.positions.iloc[-1]), 0), _pct(mix.total_return),
                             f"{(res.total_return - mix.total_return) * 100:+.2f}"])
            _table(["設定", "報酬", "最大回撤", "Sharpe", "買入次數", "平均持股", "期末持股",
                    "固定配置報酬", "差距(百分點)"], rows)
        print("\n固定配置 = 每天維持與該設定「平均持股」相同的比例在標的、其餘放現金，"
              "不挑時機；差距為正代表逢跌買進的時機有加分。")

    elif args.cmd == "synth":
        from .research import compare_series, synth_leveraged
        base = data.load_history(args.base, args.data_dir, years=args.years)
        cash = data.load_cash_rates(base["date"], args.data_dir, years=args.years)
        syn = synth_leveraged(base, cash, leverage=args.leverage, fee=args.fee)
        path = os.path.join(args.data_dir, f"{args.name}.csv")
        data.save_csv(syn, path)
        print(f"已合成 {args.name} = {args.leverage:g} 倍 {args.base}（管理費 {args.fee:.2%}、"
              f"融資成本 = {args.leverage - 1:g} × 國庫券利率），"
              f"{syn['date'].iloc[0]} ~ {syn['date'].iloc[-1]}，存到 {path}")
        if args.check:
            real = data.load_or_fetch(args.check, args.data_dir, period="max")
            c = compare_series(syn, real)
            print(f"\n與真實 {args.check} 比對（重疊期間 {c['start']} ~ {c['end']}，{c['days']} 天）：")
            _table(["", args.name, args.check],
                   [["年化報酬", _pct(c["cagr_a"]), _pct(c["cagr_b"])],
                    ["最大回撤", _pct(c["mdd_a"]), _pct(c["mdd_b"])]])
            print(f"日報酬相關係數 {c['corr']:.4f}；各年度報酬差距的中位數 "
                  f"{c['yearly_gap_median'] * 100:.2f} 個百分點")
            print("年化報酬與回撤接近、相關係數接近 1，代表合成序列可以用來推估上市前的表現。")

    elif args.cmd == "best":
        from .research import pick_by_risk, strategy_lab
        # 起點前若有更早的資料會拿來給均線暖機；沒有的話起點會往後延 sma_days 天
        raw = {t: data.load_history(t, args.data_dir, years=args.years)
               for t in args.tickers}
        prices = data.align_universe(raw)
        dates = prices[args.tickers[0]]["date"]
        cut = pd.Timestamp.today().normalize() - pd.DateOffset(years=args.years)
        first_start = int((pd.to_datetime(dates) < cut).sum())
        cash = cash_for(prices[args.tickers[0]], years=args.years)
        if isinstance(cash, float):
            cash = np.full(len(dates), cash)
        win, summ = strategy_lab(prices, cash, first_start, window=args.window_days,
                                 step=args.step_days, fee_bps=args.fee_bps,
                                 sma_days=args.sma_days, short_fee=args.short_fee)
        base = f"{args.tickers[0]} 買進持有"
        print(f"\n=== 策略實驗室：{'、'.join(args.tickers)}，起點 {win['start'].iloc[0]} ~ "
              f"{win['start'].iloc[-1]}，共 {win['start'].nunique()} 段一年期區間 ===")
        print(f"現金：^IRX 國庫券每日實際利率（平均 {np.mean(cash):.2%}）"
              if str(args.cash_rate).lower() == "auto" else f"現金：固定 {np.mean(cash):.2%}")
        rows = [[r["strategy"], _pct(r["mean"]), _pct(r["median"]), _pct(r["p10"]),
                 _pct(r["worst"]), _pct(r["worst_mdd"]),
                 "—" if r["strategy"] == base else _pct(r["beat_base"], 0),
                 _pct(r["full_cagr"]), _pct(r["full_mdd"])]
                for _, r in summ.sort_values("full_cagr", ascending=False).iterrows()]
        _table(["策略", "一年平均", "一年中位", "最差10%", "最差一年", "年內最差回撤",
                f"勝{base}", "全期年化", "全期最大回撤"], rows)

        print("\n依你能承受的最大回撤挑選（全期與任一年內的回撤都不得超過門檻；"
              "以全期年化報酬最高者為準）：")
        picks = []
        for p in pick_by_risk(summ):
            label = "不限" if p["budget"] is None else f"不超過 -{p['budget']:.0%}"
            r = p["row"]
            if r is None:
                picks.append([label, "（沒有符合的策略）", "", "", "", ""])
            else:
                picks.append([label, r["strategy"], _pct(r["full_cagr"]), _pct(r["full_mdd"]),
                              _pct(r["median"]), _pct(r["worst"])])
        _table(["可承受回撤", "最佳策略", "全期年化", "全期最大回撤", "一年中位", "最差一年"],
               picks)

        os.makedirs(args.out_dir, exist_ok=True)
        tag = "_".join(args.tickers)
        win.to_csv(os.path.join(args.out_dir, f"best_{tag}_windows.csv"), index=False,
                   encoding="utf-8-sig")
        summ.to_csv(os.path.join(args.out_dir, f"best_{tag}_summary.csv"), index=False,
                    encoding="utf-8-sig")
        print(f"\n明細已存到 {args.out_dir}{os.sep}best_{tag}_*.csv")
        print("說明：固定比例為每日再平衡、不計再平衡成本；均線濾網＝收盤在均線之上隔日持有，"
              "否則轉國庫券；均線多空＝跌破均線改做空（以反向 ETF 計：賺國庫券利率、"
              f"扣 {args.short_fee:.2%} 年費）；均線策略皆含交易成本。"
              "\n逢跌加碼：一年期欄位每段從全現金開始；全期欄位從第一個起點以全現金開始一路執行。"
              "\n「全期」＝從第一個起點一路執行到最後，最貼近長期實際執行的報酬與最大跌幅。"
              "\n挑選結果是對過去資料的最佳，不保證未來；相鄰區間高度重疊，看大方向勝過看小數點。")

    elif args.cmd == "rolling":
        from .research import by_start_year, dip_configs, rolling_dip, summarize
        configs = dip_configs(args.triggers, args.buy_fracs, args.mode,
                              args.cooldown, args.init_frac, args.exit_mode, args.tp_pct)
        focus = configs[0]["label"]
        os.makedirs(args.out_dir, exist_ok=True)
        for t in args.tickers:
            df = data.load_history(t, args.data_dir, years=args.years)
            cut = pd.Timestamp.today().normalize() - pd.DateOffset(years=args.years)
            df = df[pd.to_datetime(df["date"]) >= cut].reset_index(drop=True)
            cash = cash_for(df, years=args.years)
            if isinstance(cash, float):
                cash_note = f"固定 {cash:.2%}"
            else:
                cash_note = f"^IRX 國庫券每日實際利率（期間平均 {cash.mean():.2%}，" \
                            f"最低 {cash.min():.2%}，最高 {cash.max():.2%}）"
            win = rolling_dip(df, configs, window=args.window_days, step=args.step_days,
                              cash_rate=cash, fee_bps=args.fee_bps)
            n_win = win["start"].nunique()
            print(f"\n=== {t} 滾動測試：{df['date'].iloc[0]} ~ {df['date'].iloc[-1]}，"
                  f"每 {args.step_days} 個交易日一個起點、每段 {args.window_days} 天，共 {n_win} 段 ===")
            print(f"現金利率：{cash_note}")
            summ = summarize(win)
            rows = [[r["config"], _pct(r["mean"]), _pct(r["median"]), _pct(r["worst"]),
                     _pct(r["best"]), _pct(r["beat_bh"], 0), _pct(r["beat_mix"], 0),
                     _pct(r["median_expo"], 0), _pct(r["worst_mdd"]), _pct(r["spent"], 0)]
                    for _, r in summ.iterrows()]
            _table(["設定", "平均報酬", "中位報酬", "最差", "最好", "勝買進持有",
                    "勝固定配置", "中位持股", "最差回撤", "資金用盡"], rows)

            print(f"\n【{focus}】依起始年份（各欄為中位數報酬）：")
            yr = by_start_year(win, focus)
            _table(["起始年", "段數", "逢跌加碼", "買進持有", "固定配置", "勝買進持有"],
                   [[r["year"], str(r["n"]), _pct(r["dip"]), _pct(r["bh"]), _pct(r["mix"]),
                     _pct(r["beat_bh"], 0)] for _, r in yr.iterrows()])

            g = win[win["config"] == focus]
            print(f"\n【{focus}】報酬最差的 3 個起點：")
            _table(["期間", "逢跌加碼", "買進持有", "期末持股"],
                   [[f"{r['start']} ~ {r['end']}", _pct(r["dip_ret"]), _pct(r["bh_ret"]),
                     _pct(r["end_expo"], 0)] for _, r in g.nsmallest(3, "dip_ret").iterrows()])

            out = os.path.join(args.out_dir, f"rolling_{t}.csv")
            win.to_csv(out, index=False, encoding="utf-8-sig")
            print(f"\n每段明細已存到 {out}")
        print("\n說明：每段都從全現金開始、以該段第一天為高點；相鄰起點的區間高度重疊，"
              "勝率不是獨立樣本，看趨勢與最差情況比看精確百分比更有意義。"
              f"\n「資金用盡」= 期末持股 ≥ 98% 的段數比例；按年份與最差起點的表格以第一個設定（{focus}）為準。")

    elif args.cmd == "rotation":
        from .backtest import run_rotation
        uni = data.align_universe(load_uni(args.tickers))
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
        uni = load_uni(args.tickers)
        print(f"對 {len(uni)} 檔標的做組合退火（策略 {args.strategy}，{args.steps} 步，目標 {args.objective}）...")
        best, res, log = annealing.optimize_portfolio(
            uni, args.strategy, objective=args.objective, fee_bps=args.fee_bps,
            steps=args.steps, seed=args.seed, verbose=True)
        print("\n最佳組合權重:", json.dumps(best["weights"], ensure_ascii=False))
        _print(res.summary())


if __name__ == "__main__":
    main()
