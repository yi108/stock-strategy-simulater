"""訊號策略：每個策略輸入日 K DataFrame，輸出 0/1 持倉訊號（當日收盤決定、次日生效由回測器處理）。

每個策略附帶 PARAM_SPACE，描述可被模擬退火調整的參數範圍：
  (lo, hi, "int")    整數參數
  (lo, hi, "float")  浮點參數
"""

from __future__ import annotations

import pandas as pd

from . import indicators as ind

ParamSpace = dict[str, tuple[float, float, str]]


class Strategy:
    NAME = "base"
    PARAM_SPACE: ParamSpace = {}

    def __init__(self, **params):
        defaults = {k: v for k, v in self.default_params().items()}
        defaults.update(params)
        self.params = defaults

    @classmethod
    def default_params(cls) -> dict:
        out = {}
        for k, (lo, hi, kind) in cls.PARAM_SPACE.items():
            mid = (lo + hi) / 2
            out[k] = int(round(mid)) if kind == "int" else round(mid, 3)
        return out

    def signals(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def __repr__(self):
        p = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.NAME}({p})"


class BuyHold(Strategy):
    """買進持有，當作基準。"""
    NAME = "buyhold"

    def signals(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=df.index)


class MACDCross(Strategy):
    """MACD 快慢線黃金交叉做多、死亡交叉出場。"""
    NAME = "macd"
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
        return (m["macd"] > m["signal"]).astype(float)


class BollingerMeanReversion(Strategy):
    """布林通道均值回歸：跌破下軌買進，回到中軌（或上軌）出場。"""
    NAME = "boll_mr"
    PARAM_SPACE: ParamSpace = {
        "window": (10, 40, "int"),
        "num_std": (1.5, 3.0, "float"),
        "exit_pct_b": (0.5, 1.0, "float"),  # %b 回升到此值即出場
    }

    @classmethod
    def default_params(cls):
        return {"window": 20, "num_std": 2.0, "exit_pct_b": 0.5}

    def signals(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        b = ind.bollinger(df["close"], window=int(p["window"]), num_std=float(p["num_std"]))
        pos = 0.0
        out = []
        for pct in b["pct_b"]:
            if pd.isna(pct):
                out.append(0.0)
                continue
            if pos == 0.0 and pct < 0.0:
                pos = 1.0
            elif pos == 1.0 and pct >= float(p["exit_pct_b"]):
                pos = 0.0
            out.append(pos)
        return pd.Series(out, index=df.index)


class BollingerBreakout(Strategy):
    """布林通道突破：收盤突破上軌做多，跌破中軌出場（順勢）。"""
    NAME = "boll_bo"
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
        for c, up, mid in zip(close, b["upper"], b["mid"]):
            if pd.isna(up):
                out.append(0.0)
                continue
            if pos == 0.0 and c > up:
                pos = 1.0
            elif pos == 1.0 and c < mid:
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


STRATEGIES: dict[str, type[Strategy]] = {
    cls.NAME: cls
    for cls in [BuyHold, MACDCross, BollingerMeanReversion, BollingerBreakout, MACDBollinger]
}


def make(name: str, **params) -> Strategy:
    return STRATEGIES[name](**params)
