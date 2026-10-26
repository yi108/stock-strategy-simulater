# 股票策略模擬器 (Stock Strategy Simulator)

根據歷史日 K 報價模擬各種訊號策略（MACD、布林通道等）與投資組合，
並用**模擬退火（Simulated Annealing）**搜尋最佳的策略參數與組合權重。

## 安裝

```bash
pip install -r requirements.txt
```

只有抓真實報價需要 `yfinance`；離線示範只需 pandas + numpy。

## 網頁介面（推薦）

```bash
python -m simulator.web          # 開 http://127.0.0.1:8000
```

純標準庫，不需額外安裝。介面功能：

- **資料區**：一鍵產生示範資料，或輸入代號抓真實報價
- **自由組合**：每一列自由選「股票 × 策略 × 參數 × 權重」，
  同一檔股票可以重複出現、搭配不同策略；可複製/刪除列；
  每列都有停損/停利（sl_pct/tp_pct）與做空（short）參數
- **回測**：淨值曲線 + 回撤曲線（含等權買進持有基準、各列曲線可切換）、
  總報酬 / CAGR / Sharpe / 最大回撤卡片、每筆交易明細
- **訓練／驗證切分**：退火只在訓練期評分，驗證期在圖上以底色標示，
  訓練好、驗證爛會亮「過擬合警訊」
- **模擬退火**：即時進度條與收斂曲線、可隨時停止取目前最佳解、
  多重啟避免局部最優；跑完自動把最佳解套回表格，可手動微調再退火
- **Walk-forward 滾動驗證**：逐段退火 → 下一段樣本外驗證，
  拼出一條「每一天都是模型沒看過的資料」的淨值曲線
- **參數穩健性熱圖**：掃描任兩個參數的網格，看最佳解是高原還是孤峰；
  點熱圖可直接把該組參數套回組合列
- **動量輪動**：每期買過去表現最強的 N 檔（橫斷面排名），參數也能退火
- **實驗紀錄**：每次執行自動記錄（存在瀏覽器）、一鍵還原設定、
  勾選多筆疊圖比較淨值、可匯出/匯入 JSON

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
| `rsi` | RSI 均值回歸：超賣買進、回中性出場 | window, oversold, exit_level |
| `sma` | 均線交叉：快線上穿慢線做多 | fast, slow |
| `tsmom` | 時序動量：過去 N 日報酬為正做多 | lookback |
| `dip` | 逢跌加碼：從高點回撤達門檻就投入固定比例資金，只買不賣 | dd_trigger, buy_frac, mode, cooldown, init_frac |

除 `buyhold` 外，所有策略另有共通參數：`sl_pct`/`tp_pct` 停損停利
（0 = 關閉，收盤價近似執行）；可做空的策略（`macd_boll` 除外）有
`short`（1 = 訊號轉空時放空）。這些也都能被退火調整。

自訂參數回測：

```bash
python -m simulator.cli backtest -s macd -t AAPL -p fast=8 slow=30 signal=7
```

## 逢跌加碼研究（例：QQQ vs QLD）

`dip` 策略從全現金出發（`init_frac` 可設定第一天先投入的比例），
每當價格從期間高點回撤達 `dd_trigger` 就投入 `buy_frac` × 初始資金，只買不賣：

- `mode 0`：每多跌一階買一次（-3%、-6%、-9%…），創新高後重置
- `mode 1`：比「上次買價」再跌 `dd_trigger` 才買
- `mode 2`：只要回撤 ≥ `dd_trigger`，每 `cooldown` 個交易日買一次

```bash
python -m simulator.cli fetch -t QQQ QLD --period 1y
python -m simulator.cli dip -t QQQ QLD --triggers 0.03 0.05 0.10 --buy-fracs 0.05 0.10 --cash-rate 0.04
```

網頁上把兩列都設成 `dip`（一列 QQQ、一列 QLD），結果區的對照表會並列每列的
報酬、回撤、平均/期末持股比例，以及同一檔買進持有的報酬與回撤。
建議把「現金年利率」設成貨幣基金利率，未投入的現金才不會被當成零報酬；
熱圖選 `dd_trigger × buy_frac` 可以一次看完所有觸發幅度與投入比例的組合。

## 回測規則

- 訊號以**當日收盤**計算，**次日**才生效（`shift(1)`），避免前視偏誤
- 每次換手收取雙邊交易成本 `--fee-bps` + 滑價 `--slippage-bps`
- 停損/停利以收盤價近似：收盤觸發、隔日出場，之後維持空手直到
  原始訊號改變才重新進場
- 波動率目標（vol target）：倉位 × min(1, 目標年化波動 / 近 20 日年化波動)
- 現金年利率（`cash_rate`）：未投入的現金每日按年利率 / 252 計息（預設 0）
- 績效指標：總報酬、CAGR、年化 Sharpe、最大回撤、交易次數（實際進出場筆數）

## 模擬退火

- 指數降溫 + Metropolis 接受準則；高溫時大步跳躍、低溫時微調
- **多重啟**（restarts）：不同種子跑多輪、彼此熱啟動，取全域最佳
- **可中止**：隨時停止並取目前最佳解；進度與收斂曲線即時回報
- 目標函數可選 `sharpe`（預設）、`total_return`、
  `calmar_mix`（Sharpe 加上最大回撤懲罰）
- `optimize` 調單一策略參數；自由組合退火把
  **每列的參數 + 權重**編成同一個狀態向量一起退火

## 對抗過擬合的三件武器

在同一段歷史上退火出來的參數，是對**那段行情**的最佳解，不保證未來有效。
這個工具內建三層防線：

1. **訓練／驗證切分**（`--train-frac 0.7`）：退火只看前 70%，
   後 30% 是它沒看過的資料。訓練 Sharpe 3.3、驗證 -1.4 這種結果，
   就是在告訴你「這組參數是背答案背出來的」
2. **Walk-forward**（`walkforward` 指令 / 網頁卡片）：逐段退火、
   逐段樣本外驗證，最接近實際操作的測法
3. **參數穩健性熱圖**（網頁卡片）：好的參數是一整片高原，
   孤立尖峰就是過擬合警訊

另外：交易成本與滑價開著測，高換手策略會現形；`record` 模式記錄
未來一年的真實報價，是最終極的樣本外測試。

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
