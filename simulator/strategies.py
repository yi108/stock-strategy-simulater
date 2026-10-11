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


STRATEGIES: dict[str, type[Strategy]] = {
    cls.NAME: cls
    for cls in [BuyHold, MACDCross, BollingerMeanReversion, BollingerBreakout,
                MACDBollinger, RSIReversion, SMACross, TSMomentum]
}


def make(name: str, **params) -> Strategy:
    return STRATEGIES[name](**params)
