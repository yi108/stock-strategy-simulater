"""訊號策略：每個策略輸入日 K DataFrame，輸出 -1/0/+1 持倉訊號
（-1 做空、0 空手、+1 做多；當日收盤決定、次日生效由回測器處理）。

每個策略附帶 PARAM_SPACE，描述可被模擬退火調整的參數範圍：
  (lo, hi, "int")    整數參數
  (lo, hi, "float")  浮點參數

除了各自的指標參數，所有策略（買進持有除外）還有共通的風控參數：
  sl_pct / tp_pct  停損 / 停利（0 = 關閉；由回測器以收盤價近似執行）
  short            1 = 允許做空（CAN_SHORT 的策略才有）
"""

from __future__ import annotations

import pandas as pd

from . import indicators as ind

ParamSpace = dict[str, tuple[float, float, str]]

RISK_SPACE: ParamSpace = {
    "sl_pct": (0.0, 0.3, "float"),   # 停損：虧損達此比例出場
    "tp_pct": (0.0, 0.6, "float"),   # 停利：獲利達此比例出場
}
SHORT_SPACE: ParamSpace = {"short": (0, 1, "int")}


class Strategy:
    NAME = "base"
    PARAM_SPACE: ParamSpace = {}
    CAN_SHORT = False
    HAS_RISK = True   # 是否附帶 sl/tp 風控參數

    def __init__(self, **params):
        merged = self.full_defaults()
        for k, v in params.items():
            if k in merged:
                merged[k] = v
        self.params = merged

    @classmethod
    def full_space(cls) -> ParamSpace:
        sp = dict(cls.PARAM_SPACE)
        if cls.HAS_RISK:
            sp.update(RISK_SPACE)
        if cls.CAN_SHORT:
            sp.update(SHORT_SPACE)
        return sp

    @classmethod
    def default_params(cls) -> dict:
        out = {}
        for k, (lo, hi, kind) in cls.PARAM_SPACE.items():
            mid = (lo + hi) / 2
            out[k] = int(round(mid)) if kind == "int" else round(mid, 3)
        return out

    @classmethod
    def full_defaults(cls) -> dict:
        d = dict(cls.default_params())
        if cls.HAS_RISK:
            d.setdefault("sl_pct", 0.0)
            d.setdefault("tp_pct", 0.0)
        if cls.CAN_SHORT:
            d.setdefault("short", 0)
        return d

    @property
    def allow_short(self) -> bool:
        return bool(int(self.params.get("short", 0)))

    def signals(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def __repr__(self):
        p = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.NAME}({p})"


class BuyHold(Strategy):
    """買進持有，當作基準。"""
    NAME = "buyhold"
    HAS_RISK = False

    def signals(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=df.index)


class MACDCross(Strategy):
    """MACD 黃金交叉做多、死亡交叉出場（short=1 時死叉改做空）。"""
    NAME = "macd"
    CAN_SHORT = True
    PARAM_SPACE: ParamSpace = {
        "fast": (5, 20, "int"),
        "slow": (21, 60, "int"),
        "signal": (5, 15, "int"),
    }

    @classmethod
    def default_params(cls):
        return {"fast": 12, "slow": 26, "signal": 9}

    def signals(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        fast, slow = sorted((int(p["fast"]), int(p["slow"])))
        if fast == slow:
            slow += 1
        m = ind.macd(df["close"], fast=fast, slow=slow, signal=int(p["signal"]))
        bull = (m["macd"] > m["signal"]).astype(float)
        if self.allow_short:
            return bull * 2 - 1.0   # 多頭 +1、空頭 -1
        return bull


class BollingerMeanReversion(Strategy):
    """布林通道均值回歸：跌破下軌買進，%b 回升出場；short=1 時突破上軌放空。"""
    NAME = "boll_mr"
    CAN_SHORT = True
    PARAM_SPACE: ParamSpace = {
        "window": (10, 40, "int"),
        "num_std": (1.5, 3.0, "float"),
        "exit_pct_b": (0.5, 1.0, "float"),  # 多單：%b 回升到此值出場；空單鏡像
    }

    @classmethod
    def default_params(cls):
        return {"window": 20, "num_std": 2.0, "exit_pct_b": 0.5}

    def signals(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        b = ind.bollinger(df["close"], window=int(p["window"]), num_std=float(p["num_std"]))
        exit_b = float(p["exit_pct_b"])
        pos = 0.0
        out = []
        for pct in b["pct_b"]:
            if pd.isna(pct):
                out.append(0.0)
                continue
            if pos == 0.0:
                if pct < 0.0:
                    pos = 1.0
                elif self.allow_short and pct > 1.0:
                    pos = -1.0
            elif pos == 1.0 and pct >= exit_b:
                pos = 0.0
            elif pos == -1.0 and pct <= 1.0 - exit_b:
                pos = 0.0
            out.append(pos)
        return pd.Series(out, index=df.index)


class BollingerBreakout(Strategy):
    """布林通道突破：突破上軌做多、跌破中軌出場；short=1 時跌破下軌放空。"""
    NAME = "boll_bo"
    CAN_SHORT = True
    PARAM_SPACE: ParamSpace = {
        "window": (10, 40, "int"),
        "num_std": (1.0, 2.5, "float"),
    }

    @classmethod
    def default_params(cls):
        return {"window": 20, "num_std": 2.0}

    def signals(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        b = ind.bollinger(df["close"], window=int(p["window"]), num_std=float(p["num_std"]))
        close = df["close"]
        pos = 0.0
        out = []
        for c, up, mid, lo in zip(close, b["upper"], b["mid"], b["lower"]):
            if pd.isna(up):
                out.append(0.0)
                continue
            if pos == 0.0:
                if c > up:
                    pos = 1.0
                elif self.allow_short and c < lo:
                    pos = -1.0
            elif pos == 1.0 and c < mid:
                pos = 0.0
            elif pos == -1.0 and c > mid:
                pos = 0.0
            out.append(pos)
        return pd.Series(out, index=df.index)


class MACDBollinger(Strategy):
    """組合訊號：MACD 多頭 且 %b 低於門檻才進場，MACD 轉空或 %b 過熱出場。"""
    NAME = "macd_boll"
    PARAM_SPACE: ParamSpace = {
        "fast": (5, 20, "int"),
        "slow": (21, 60, "int"),
        "signal": (5, 15, "int"),
        "window": (10, 40, "int"),
        "num_std": (1.5, 3.0, "float"),
        "entry_pct_b": (0.2, 0.9, "float"),
        "exit_pct_b": (0.9, 1.3, "float"),
    }

    @classmethod
    def default_params(cls):
        return {"fast": 12, "slow": 26, "signal": 9,
                "window": 20, "num_std": 2.0,
                "entry_pct_b": 0.6, "exit_pct_b": 1.05}

    def signals(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        fast, slow = sorted((int(p["fast"]), int(p["slow"])))
        if fast == slow:
            slow += 1
        m = ind.macd(df["close"], fast=fast, slow=slow, signal=int(p["signal"]))
        b = ind.bollinger(df["close"], window=int(p["window"]), num_std=float(p["num_std"]))
        bull = m["macd"] > m["signal"]
        pos = 0.0
        out = []
        for is_bull, pct in zip(bull, b["pct_b"]):
            if pd.isna(pct):
                out.append(0.0)
                continue
            if pos == 0.0 and is_bull and pct < float(p["entry_pct_b"]):
                pos = 1.0
            elif pos == 1.0 and (not is_bull or pct > float(p["exit_pct_b"])):
                pos = 0.0
            out.append(pos)
        return pd.Series(out, index=df.index)


class RSIReversion(Strategy):
    """RSI 均值回歸：超賣買進、回到中性出場；short=1 時超買放空（鏡像門檻）。"""
    NAME = "rsi"
    CAN_SHORT = True
    PARAM_SPACE: ParamSpace = {
        "window": (5, 30, "int"),
        "oversold": (15, 40, "float"),    # 多單進場門檻；空單用 100-oversold
        "exit_level": (45, 70, "float"),  # 多單出場門檻；空單用 100-exit_level
    }

    @classmethod
    def default_params(cls):
        return {"window": 14, "oversold": 30.0, "exit_level": 55.0}

    def signals(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        r = ind.rsi(df["close"], window=int(p["window"]))
        lo = float(p["oversold"])
        ex = float(p["exit_level"])
        pos = 0.0
        out = []
        for v in r:
            if pd.isna(v):
                out.append(0.0)
                continue
            if pos == 0.0:
                if v < lo:
                    pos = 1.0
                elif self.allow_short and v > 100 - lo:
                    pos = -1.0
            elif pos == 1.0 and v >= ex:
                pos = 0.0
            elif pos == -1.0 and v <= 100 - ex:
                pos = 0.0
            out.append(pos)
        return pd.Series(out, index=df.index)


class SMACross(Strategy):
    """均線交叉：快線上穿慢線做多；short=1 時下穿改做空。"""
    NAME = "sma"
    CAN_SHORT = True
    PARAM_SPACE: ParamSpace = {
        "fast": (5, 60, "int"),
        "slow": (20, 150, "int"),
    }

    @classmethod
    def default_params(cls):
        return {"fast": 20, "slow": 60}

    def signals(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        fast, slow = sorted((int(p["fast"]), int(p["slow"])))
        if fast == slow:
            slow += 1
        f = ind.sma(df["close"], fast)
        s = ind.sma(df["close"], slow)
        bull = (f > s).astype(float)
        bull[s.isna()] = 0.0
        if self.allow_short:
            sig = bull * 2 - 1.0
            sig[s.isna()] = 0.0
            return sig
        return bull


class TSMomentum(Strategy):
    """時序動量：過去 lookback 日報酬為正做多；short=1 時為負做空。"""
    NAME = "tsmom"
    CAN_SHORT = True
    PARAM_SPACE: ParamSpace = {
        "lookback": (20, 180, "int"),
    }

    @classmethod
    def default_params(cls):
        return {"lookback": 120}

    def signals(self, df: pd.DataFrame) -> pd.Series:
        lb = int(self.params["lookback"])
        mom = df["close"].pct_change(lb)
        sig = (mom > 0).astype(float)
        if self.allow_short:
            sig = sig - (mom < 0).astype(float)
        sig[mom.isna()] = 0.0
        return sig


class DipBuy(Strategy):
    """逢跌加碼：從高點回撤達 dd_trigger 就投入 buy_frac 的初始資金，只買不賣（mode 0 每跌一階買一次 / 1 從上次買價再跌一階 / 2 回撤中每 cooldown 天買一次）。

    與訊號策略不同，這是以「現金 + 股數」逐筆累積的模擬：
    - 初始資金 1.0，init_frac 為第一天就先投入的比例，其餘為現金
    - 高點 = 回測期間至今的最高收盤價；創新高時重置加碼階梯
    - 每次加碼金額 = buy_frac × 初始資金（現金不足時用剩下的全部）
    - 於觸發當日收盤成交（等同收盤前下單），只收買進成本
    - mode 0：回撤每跨過一個 dd_trigger 的整數倍買一次（-3%、-6%、-9%…）
      mode 1：價格比「上次買價」（或高點）再跌 dd_trigger 才買
      mode 2：只要回撤 ≥ dd_trigger，每隔 cooldown 個交易日買一次
    """
    NAME = "dip"
    HAS_RISK = False
    PARAM_SPACE: ParamSpace = {
        "dd_trigger": (0.01, 0.20, "float"),
        "buy_frac": (0.01, 0.50, "float"),
        "mode": (0, 2, "int"),
        "cooldown": (1, 20, "int"),
        "init_frac": (0.0, 1.0, "float"),
    }

    @classmethod
    def default_params(cls):
        return {"dd_trigger": 0.03, "buy_frac": 0.05, "mode": 0,
                "cooldown": 5, "init_frac": 0.0}

    def simulate(self, df: pd.DataFrame, fee_bps: float = 10.0,
                 slippage_bps: float = 0.0, cash_rate: float = 0.0) -> dict:
        p = self.params
        trig = max(1e-4, float(p["dd_trigger"]))
        lot = max(0.0, float(p["buy_frac"]))
        mode = int(p["mode"])
        cooldown = max(1, int(p["cooldown"]))
        cost = (fee_bps + slippage_bps) / 10_000.0
        daily_cash = cash_rate / 252.0

        close = df["close"].to_numpy(dtype=float)
        dates = df["date"].astype(str).to_numpy()
        n = len(close)
        cash, shares = 1.0, 0.0
        lots: list[tuple[int, float, float]] = []   # (日索引, 投入金額, 回撤)

        def buy(i: int, amount: float, dd: float) -> None:
            nonlocal cash, shares
            amount = min(amount, cash)
            if amount <= 1e-9:
                return
            shares += amount * (1.0 - cost) / close[i]
            cash -= amount
            lots.append((i, amount, dd))

        peak = close[0] if n else 0.0
        rung = 0
        ref_price = None
        last_buy_i = -10**9
        equity = []
        exposure = []
        for i in range(n):
            c = close[i]
            if i > 0:
                cash *= 1.0 + daily_cash
            if i == 0 and float(p["init_frac"]) > 0:
                buy(0, float(p["init_frac"]), 0.0)
            if c >= peak:
                peak, rung, ref_price = c, 0, None
            dd = 1.0 - c / peak
            if mode == 0:
                k = int(dd / trig + 1e-9)
                if k > rung:
                    buy(i, lot * (k - rung), dd)
                    rung = k
            elif mode == 1:
                ref = ref_price if ref_price is not None else peak
                if c <= ref * (1.0 - trig):
                    buy(i, lot, dd)
                    ref_price = c
            else:
                if dd >= trig and i - last_buy_i >= cooldown:
                    buy(i, lot, dd)
                    last_buy_i = i
            eq = cash + shares * c
            equity.append(eq)
            exposure.append(shares * c / eq if eq > 0 else 0.0)

        last = n - 1
        trades_log = [{
            "entry": dates[i], "exit": dates[last],
            "side": f"加碼 {amt:.1%}（回撤 {dd:.1%}）",
            "days": last - i,
            "ret": round(float(close[last] / close[i] * (1.0 - cost) - 1.0), 4),
            "open": True,
        } for i, amt, dd in lots]
        return {"equity": equity, "exposure": exposure, "trades_log": trades_log,
                "cash_left": cash}

    def signals(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(self.simulate(df)["exposure"], index=df.index)


STRATEGIES: dict[str, type[Strategy]] = {
    cls.NAME: cls
    for cls in [BuyHold, MACDCross, BollingerMeanReversion, BollingerBreakout,
                MACDBollinger, RSIReversion, SMACross, TSMomentum, DipBuy]
}


def make(name: str, **params) -> Strategy:
    return STRATEGIES[name](**params)
