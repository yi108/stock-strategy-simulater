# 股票策略模擬器 (Stock Strategy Simulator)

根據歷史日 K 報價模擬各種訊號策略（MACD、布林通道等）與投資組合，
並用**模擬退火（Simulated Annealing）**搜尋最佳的策略參數與組合權重。

## 安裝

```bash
pip install -r requirements.txt
```

只有抓真實報價需要 `yfinance`；離線示範只需 pandas + numpy。

## 快速開始（離線示範）

```bash
# 1. 產生 5 檔合成示範資料（約一年日 K）到 data/
python -m simulator.cli sample

# 2. 在同一檔標的上比較所有策略
python -m simulator.cli compare -t DEMO_A

# 3. 模擬退火找 MACD 最佳參數
python -m simulator.cli optimize -s macd -t DEMO_A --steps 400

# 4. 模擬退火同時最佳化「組合權重 + 各標的策略參數」
python -m simulator.cli optimize-portfolio -s macd_boll --steps 600
```

## 使用真實報價

在可連外網的電腦上：

```bash
# 抓過去一年的日 K（美股、台股皆可，台股代號如 2330.TW）
python -m simulator.cli fetch -t AAPL MSFT 2330.TW --period 1y

# 之後所有回測/退火指令直接用這些 ticker
python -m simulator.cli compare -t AAPL
python -m simulator.cli optimize -s boll_mr -t 2330.TW
```

### 記錄接下來一年的報價

`record` 會抓最近幾天的報價並 append 到既有 CSV（自動去重）。
每天收盤後跑一次（可排 cron），一年後就累積出完整的未來一年資料：

```bash
python -m simulator.cli record -t AAPL MSFT 2330.TW
# crontab 範例（台股收盤後 15:00）
# 0 15 * * 1-5 cd /path/to/repo && python -m simulator.cli record -t 2330.TW
```

也可以自備 CSV 放進 `data/<TICKER>.csv`，格式：
`date,open,high,low,close,volume`（date 為 YYYY-MM-DD）。

## 內建策略

| 名稱 | 說明 | 可退火參數 |
|---|---|---|
| `buyhold` | 買進持有（基準） | — |
| `macd` | MACD 黃金交叉做多、死亡交叉出場 | fast, slow, signal |
| `boll_mr` | 布林通道均值回歸：跌破下軌買、回到中軌賣 | window, num_std, exit_pct_b |
| `boll_bo` | 布林通道突破：突破上軌買、跌破中軌賣 | window, num_std |
| `macd_boll` | MACD 多頭 + %b 位置過濾的組合訊號 | 上述全部 + entry/exit_pct_b |

自訂參數回測：

```bash
python -m simulator.cli backtest -s macd -t AAPL -p fast=8 slow=30 signal=7
```

## 回測規則

- 訊號以**當日收盤**計算，**次日**才生效（`shift(1)`），避免前視偏誤
- 每次換手收取雙邊交易成本 `--fee-bps`（預設 10 bps = 0.1%）
- 績效指標：總報酬、CAGR、年化 Sharpe、最大回撤、交易次數

## 模擬退火

- 指數降溫 + Metropolis 接受準則；高溫時大步跳躍、低溫時微調
- 目標函數可選 `--objective sharpe`（預設）、`total_return`、
  `calmar_mix`（Sharpe 加上最大回撤懲罰）
- `optimize` 只調單一策略參數；`optimize-portfolio` 把
  **每檔標的的參數 + 組合權重**編成同一個狀態向量一起退火

## ⚠️ 過擬合警告

在同一段歷史資料上退火出來的參數，是對**那段行情**的最佳解，
不保證未來有效（這正是 `record` 模式的用途：先用過去一年資料
找參數，再用記錄下來的下一年資料做樣本外驗證）。建議：

1. 把資料切成訓練段/驗證段，退火只看訓練段
2. 偏好參數變動時績效穩定的解，而不是單點最優
3. 交易成本開著測（`--fee-bps`），高換手策略會現形

## 專案結構

```
simulator/
  data.py        # 抓取(yfinance/stooq)、記錄(append)、CSV、合成資料
  indicators.py  # SMA/EMA/MACD/布林通道/RSI
  strategies.py  # 訊號策略與參數空間定義
  backtest.py    # 單標的 + 組合回測引擎、績效指標
  annealing.py   # 通用模擬退火 + 參數/組合最佳化
  cli.py         # 命令列介面
```

新增策略：在 `strategies.py` 繼承 `Strategy`，實作 `signals()`，
宣告 `PARAM_SPACE`，加進 `STRATEGIES` 即可被回測與退火使用。
