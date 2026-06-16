# -*- coding: utf-8 -*-
"""
台股動能飆股篩選器 (Taiwan Momentum Breakout Screener)
=====================================================

設計哲學：融合 William O'Neil (CANSLIM)、Jesse Livermore (關鍵點突破) 的
「主升段飆股」特徵 — 創新高 + 量增 + 均線多頭排列 + 高相對強度(RS)，
再疊加三大法人籌碼面與新聞/總經情緒，給出 0~100 的信心分數。

核心原則：寧缺勿濫。只輸出超過信心閥值的標的，最多 20 檔；
若當天盤勢不佳，誠實輸出實際檔數（可能 5 檔、甚至 0 檔），絕不硬湊。

適用環境：iPhone 上的 Juno / a-Shell / Pyto，亦可在桌機執行。
相依套件：yfinance, pandas, numpy, requests（皆免金鑰）。

使用方式：
    python tw_momentum_screener.py            # 跑今日選股
    python tw_momentum_screener.py --backtest # 跑歷史回測

作者：量化策略範本，僅供研究教育用途，非投資建議。
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import quote

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:  # 讓程式在未裝套件時給出清楚提示
    print("缺少 yfinance，請先執行：!pip install yfinance")
    raise

try:
    import requests
except ImportError:
    requests = None


# =============================================================================
# 0. 參數設定（集中管理，方便手機上快速調整）
# =============================================================================

@dataclass
class Config:
    # --- 輸出控管 ---
    max_picks: int = 20            # 每日最多輸出檔數
    top_n: int = 3                 # 首選推薦檔數
    confidence_threshold: float = 70.0  # 信心閥值；達不到就不輸出（寧缺勿濫）

    # --- 資料抓取 ---
    lookback_days: int = 260       # 抓約一年交易日，足夠算 52 週高點與 RS
    index_symbol: str = "^TWII"    # 加權指數，用來算相對強度 RS

    # --- 技術門檻（硬性過濾，未過直接淘汰）---
    min_price: float = 10.0        # 排除雞蛋水餃股
    min_avg_volume: int = 1_000_000  # 20 日均量（股數）門檻，確保流動性
    volume_surge_mult: float = 1.5   # 當日量 > 1.5 倍 20 日均量 才算爆量
    near_high_pct: float = 0.90      # 收盤須站上 52 週高點的 90% 以上

    # --- 評分權重（總和 = 1.0）---
    # 設計理念見 README：飆股的本質是「相對強度 + 突破 + 量能」
    w_rs: float = 0.25             # 相對強度 RS（O'Neil 最重視）
    w_breakout: float = 0.25       # 突破力道（創新高 + 爆量）
    w_trend: float = 0.20          # 均線多頭排列（趨勢確認）
    w_momentum: float = 0.15       # MACD / RSI 動能
    w_chips: float = 0.10          # 三大法人籌碼面
    w_sentiment: float = 0.05      # 新聞 / 總經情緒

    # --- 股票池 ---
    use_full_universe: bool = False   # True = 抓全上市櫃；False = 用 DEFAULT_UNIVERSE
    max_universe: int = 250        # 全市場模式下，預篩後最多保留幾檔（控制手機運算量）

    # --- 新聞情緒 ---
    fetch_sentiment: bool = True   # 是否抓新聞情緒（只針對入選 shortlist，省流量）
    sentiment_shortlist: int = 30  # 只對前 N 名候選抓新聞

    # --- 回測 ---
    bt_hold_days: int = 5          # 訊號後持有天數
    bt_target_return: float = 0.03 # 視為「成功」的報酬門檻（5 日 +3%）


CFG = Config()


# 預設股票池：台灣 50 + 中型 100 常見成分與熱門題材股。
# 可自行增減；代碼後綴 .TW（上市）或 .TWO（上櫃）。
DEFAULT_UNIVERSE = [
    "2330.TW", "2317.TW", "2454.TW", "2308.TW", "2382.TW", "2303.TW",
    "3711.TW", "2412.TW", "2881.TW", "2882.TW", "2891.TW", "1303.TW",
    "1301.TW", "2002.TW", "2603.TW", "2609.TW", "2615.TW", "3008.TW",
    "3034.TW", "3037.TW", "3231.TW", "2376.TW", "2377.TW", "2379.TW",
    "3661.TW", "3443.TW", "4966.TW", "5269.TW", "6669.TW", "8046.TW",
    "2357.TW", "2356.TW", "2049.TW", "1519.TW", "1513.TW", "6505.TW",
    "2207.TW", "9910.TW", "2105.TW", "1216.TW", "2912.TW", "2884.TW",
    "5483.TWO", "6182.TWO", "8069.TWO", "5274.TWO", "3260.TWO", "6488.TWO",
]


# =============================================================================
# 1. 技術指標（純 pandas/numpy 實作，免裝 TA-Lib，手機可跑）
# =============================================================================

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


# =============================================================================
# 2. 資料抓取
# =============================================================================

def _to_float(x) -> Optional[float]:
    """把含逗號/破折號的字串安全轉 float；無效回 None。"""
    try:
        v = str(x).replace(",", "").strip()
        if v in ("", "--", "---", "N/A"):
            return None
        return float(v)
    except (ValueError, TypeError):
        return None


def fetch_twse_all_day() -> list[dict]:
    """
    抓上市所有個股「當日」收盤與成交量（TWSE OpenAPI，免金鑰，單一請求）。
    回傳 [{symbol, name, close, volume}]。失敗回 []。
    用途：建立全市場股票池並做預篩，避免對上千檔逐一抓歷史。
    """
    if requests is None:
        return []
    url = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
    try:
        r = requests.get(url, timeout=15)
        rows = r.json()
        out = []
        for row in rows:
            code = str(row.get("Code", "")).strip()
            if not (code.isdigit() and len(code) == 4):  # 只留 4 位數普通股
                continue
            close = _to_float(row.get("ClosingPrice"))
            vol = _to_float(row.get("TradeVolume"))
            if close is None or vol is None:
                continue
            out.append({"symbol": f"{code}.TW", "name": row.get("Name", ""),
                        "close": close, "volume": vol})
        print(f"上市全市場：取得 {len(out)} 檔當日資料。")
        return out
    except Exception as e:
        print(f"上市清單抓取失敗：{e}")
        return []


def fetch_tpex_all_day() -> list[dict]:
    """抓上櫃所有個股當日收盤與量（櫃買 OpenAPI）。回傳同 fetch_twse_all_day 格式。"""
    if requests is None:
        return []
    url = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes"
    try:
        r = requests.get(url, timeout=15)
        rows = r.json()
        out = []
        for row in rows:
            code = str(row.get("SecuritiesCompanyCode", "")).strip()
            if not (code.isdigit() and len(code) == 4):
                continue
            close = _to_float(row.get("Close"))
            vol = _to_float(row.get("TradingShares"))
            if close is None or vol is None:
                continue
            out.append({"symbol": f"{code}.TWO", "name": row.get("CompanyName", ""),
                        "close": close, "volume": vol})
        print(f"上櫃全市場：取得 {len(out)} 檔當日資料。")
        return out
    except Exception as e:
        print(f"上櫃清單抓取失敗：{e}")
        return []


def build_universe() -> list[str]:
    """
    建立股票池。CFG.use_full_universe=True 時抓全上市櫃並依價/量預篩，
    取流動性最高的前 max_universe 檔（大幅減少 yfinance 歷史抓取量）。
    任何失敗都降級回 DEFAULT_UNIVERSE。
    """
    if not CFG.use_full_universe:
        return DEFAULT_UNIVERSE

    allrows = fetch_twse_all_day() + fetch_tpex_all_day()
    if not allrows:
        print("全市場清單不可用，降級為內建股票池。")
        return DEFAULT_UNIVERSE

    # 預篩：股價門檻 + 今日成交量門檻（先濾掉水餃股/冷門股）
    cands = [r for r in allrows
             if r["close"] >= CFG.min_price and r["volume"] >= CFG.min_avg_volume]
    # 依當日成交量排序，取流動性最高的前 N 檔
    cands.sort(key=lambda r: r["volume"], reverse=True)
    picked = [r["symbol"] for r in cands[: CFG.max_universe]]
    print(f"全市場預篩後股票池：{len(picked)} 檔（上限 {CFG.max_universe}）。")
    return picked or DEFAULT_UNIVERSE


def fetch_prices(symbols: list[str], lookback_days: int) -> dict[str, pd.DataFrame]:
    """批次抓取盤後日K。回傳 {symbol: DataFrame(OHLCV)}。失敗的個股自動略過。"""
    end = dt.date.today() + dt.timedelta(days=1)
    start = end - dt.timedelta(days=int(lookback_days * 1.6))  # 多抓涵蓋假日
    out: dict[str, pd.DataFrame] = {}

    print(f"抓取 {len(symbols)} 檔股價（{start} ~ {end}）...")
    try:
        raw = yf.download(
            symbols, start=start, end=end, group_by="ticker",
            auto_adjust=True, progress=False, threads=True,
        )
    except Exception as e:
        print(f"批次下載失敗，改逐檔抓取：{e}")
        raw = None

    for sym in symbols:
        try:
            if raw is not None and isinstance(raw.columns, pd.MultiIndex) and sym in raw.columns.levels[0]:
                df = raw[sym].dropna()
            else:
                df = yf.download(sym, start=start, end=end, auto_adjust=True, progress=False)
            if df is None or len(df) < 60:
                continue
            df = df.rename(columns=str.title)[["Open", "High", "Low", "Close", "Volume"]]
            out[sym] = df.dropna()
        except Exception:
            continue
    print(f"成功取得 {len(out)} 檔。")
    return out


def fetch_index(lookback_days: int) -> Optional[pd.Series]:
    """抓加權指數收盤，用於計算相對強度 RS。"""
    end = dt.date.today() + dt.timedelta(days=1)
    start = end - dt.timedelta(days=int(lookback_days * 1.6))
    try:
        idx = yf.download(CFG.index_symbol, start=start, end=end,
                          auto_adjust=True, progress=False)
        if idx is None or idx.empty:
            return None
        return idx["Close"].dropna().squeeze()
    except Exception:
        return None


def fetch_institutional_netbuy() -> dict[str, float]:
    """
    抓三大法人「個股」買賣超（證交所開放資料，免金鑰）。
    回傳 {股號(不含後綴): 買賣超張數}。任何失敗都回傳 {} 並降級。
    資料來源：TWSE T86 報表（盤後約 15:30 後更新）。
    """
    if requests is None:
        return {}
    url = "https://www.twse.com.tw/rwd/zh/fund/T86"
    today = dt.date.today().strftime("%Y%m%d")
    params = {"date": today, "selectType": "ALL", "response": "json"}
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        if data.get("stat") != "OK":
            return {}
        rows = data.get("data", [])
        result: dict[str, float] = {}
        for row in rows:
            code = row[0].strip()
            # 最後一欄通常是「三大法人買賣超股數」
            net = row[-1].replace(",", "")
            try:
                result[code] = float(net) / 1000.0  # 股 -> 張
            except ValueError:
                continue
        # 併入上櫃法人資料（櫃買中心）
        result.update(_fetch_tpex_netbuy())
        print(f"取得三大法人買賣超 {len(result)} 檔（含上市櫃）。")
        return result
    except Exception as e:
        print(f"法人資料抓取失敗（降級為純價量）：{e}")
        return _fetch_tpex_netbuy()


def _fetch_tpex_netbuy() -> dict[str, float]:
    """抓上櫃三大法人買賣超（櫃買 OpenAPI）。回傳 {股號: 張數}，失敗回 {}。"""
    if requests is None:
        return {}
    url = "https://www.tpex.org.tw/openapi/v1/tpex_3insti_daily_trading"
    try:
        r = requests.get(url, timeout=10)
        rows = r.json()
        out: dict[str, float] = {}
        for row in rows:
            code = str(row.get("SecuritiesCompanyCode", "")).strip()
            if not (code.isdigit() and len(code) == 4):
                continue
            # 欄名各版本略有差異，盡量找「三大法人買賣超」總計欄
            net = None
            for k in ("TotalBuySellShares", "TotalInstitutionalInvestorsNetBuySell",
                      "ThreeInstitutionalInvestorsNetBuySell"):
                if k in row:
                    net = _to_float(row[k])
                    break
            if net is None:
                continue
            out[code] = net / 1000.0
        return out
    except Exception:
        return {}


# 新聞情緒：正/負面關鍵字詞典（可自行擴充）。輕量、免金鑰、可離線運作。
_POS_WORDS = ["大漲", "漲停", "創新高", "獲利", "成長", "利多", "看好", "強勢",
              "突破", "加碼", "買超", "訂單", "擴產", "報喜", "優於預期", "題材",
              "受惠", "回升", "樂觀", "飆", "噴出", "法說亮眼"]
_NEG_WORDS = ["大跌", "跌停", "創新低", "虧損", "衰退", "利空", "看壞", "弱勢",
              "跌破", "賣超", "減產", "下修", "不如預期", "示警", "違約", "掏空",
              "認列損失", "悲觀", "崩", "重挫", "裁員"]


def _score_headlines(titles: list[str]) -> float:
    """以關鍵字詞典對新聞標題計分，回傳 -1 ~ +1。無標題回 0。"""
    if not titles:
        return 0.0
    pos = sum(t.count(w) for t in titles for w in _POS_WORDS)
    neg = sum(t.count(w) for t in titles for w in _NEG_WORDS)
    if pos + neg == 0:
        return 0.0
    return float(np.clip((pos - neg) / (pos + neg), -1.0, 1.0))


def fetch_news_sentiment(symbols: list[str], names: dict[str, str] | None = None) -> dict[str, float]:
    """
    新聞情緒分數 (-1 ~ +1)。用 Google News RSS 抓每檔近期標題，再以關鍵字詞典計分。
    免金鑰；任何失敗都降級為中性 0.0。建議只對 shortlist 呼叫以省流量。
    """
    names = names or {}
    out: dict[str, float] = {s: 0.0 for s in symbols}
    if requests is None or not CFG.fetch_sentiment:
        return out

    for sym in symbols:
        code = sym.split(".")[0]
        query = quote(f"{names.get(sym, code)} {code} 股")
        url = (f"https://news.google.com/rss/search?q={query}"
               "&hl=zh-TW&gl=TW&ceid=TW:zh-Hant")
        try:
            r = requests.get(url, timeout=8)
            root = ET.fromstring(r.content)
            titles = [t.text or "" for t in root.iter("title")][1:9]  # 跳過 channel 標題
            out[sym] = _score_headlines(titles)
        except Exception:
            out[sym] = 0.0  # 降級中性
    return out


# =============================================================================
# 3. 特徵計算與評分
# =============================================================================

@dataclass
class StockScore:
    symbol: str
    confidence: float
    close: float
    rs_score: float = 0.0
    breakout_score: float = 0.0
    trend_score: float = 0.0
    momentum_score: float = 0.0
    chips_score: float = 0.0
    sentiment_score: float = 0.0
    reasons: list[str] = field(default_factory=list)


def _rel_strength_pct(stock_close: pd.Series, index_close: Optional[pd.Series]) -> float:
    """
    O'Neil 式相對強度：個股近 3/6/12 月報酬，加權後相對大盤。
    回傳 0~100 的原始強度（之後再做全市場排名）。
    """
    def ret(series, n):
        if len(series) <= n:
            return 0.0
        return float(series.iloc[-1] / series.iloc[-n] - 1.0)

    # 多週期加權（近期權重高），近似 IBD RS Rating 概念
    r = 0.4 * ret(stock_close, 63) + 0.3 * ret(stock_close, 126) + 0.3 * ret(stock_close, 252)
    if index_close is not None and len(index_close) > 252:
        ri = 0.4 * ret(index_close, 63) + 0.3 * ret(index_close, 126) + 0.3 * ret(index_close, 252)
        r = r - ri  # 相對大盤的超額報酬
    return r


def compute_features(
    symbol: str,
    df: pd.DataFrame,
    index_close: Optional[pd.Series],
    chips: dict[str, float],
    sentiment: dict[str, float],
) -> Optional[StockScore]:
    """對單一個股計算所有子分數，並做硬性過濾。未過濾條件回傳 None。"""
    if len(df) < 60:
        return None

    close = df["Close"]
    high = df["High"]
    vol = df["Volume"]
    last_close = float(close.iloc[-1])
    last_vol = float(vol.iloc[-1])

    # --- 硬性過濾（寧缺勿濫第一道關卡）---
    avg_vol20 = float(vol.tail(20).mean())
    if last_close < CFG.min_price:
        return None
    if avg_vol20 < CFG.min_avg_volume:
        return None

    # 52 週高點（約 252 日）
    high_52w = float(high.tail(252).max())
    if last_close < high_52w * CFG.near_high_pct:
        return None  # 離高點太遠，不是主升段候選

    reasons: list[str] = []

    # --- (a) 相對強度 RS（原始值，稍後排名標準化）---
    rs_raw = _rel_strength_pct(close, index_close)

    # --- (b) 突破力道：創新高 + 爆量 ---
    breakout = 0.0
    vol_ratio = last_vol / avg_vol20 if avg_vol20 > 0 else 0.0
    pct_of_high = last_close / high_52w
    if pct_of_high >= 0.999:                       # 收盤創 52 週新高
        breakout += 60; reasons.append("創52週新高")
    elif pct_of_high >= 0.97:
        breakout += 35; reasons.append("逼近52週高點")
    if vol_ratio >= CFG.volume_surge_mult:         # 爆量
        breakout += 40; reasons.append(f"爆量({vol_ratio:.1f}倍均量)")
    elif vol_ratio >= 1.2:
        breakout += 20
    breakout = min(breakout, 100.0)

    # --- (c) 均線多頭排列：價 > 20MA > 60MA > 120MA ---
    ma20 = close.rolling(20).mean().iloc[-1]
    ma60 = close.rolling(60).mean().iloc[-1]
    ma120 = close.rolling(120).mean().iloc[-1] if len(close) >= 120 else ma60
    trend = 0.0
    if last_close > ma20 > ma60:
        trend += 60; reasons.append("均線多頭排列")
    elif last_close > ma20:
        trend += 30
    if ma60 > ma120:
        trend += 40
    trend = min(trend, 100.0)

    # --- (d) MACD / RSI 動能 ---
    _, _, hist = macd(close)
    rsi14 = rsi(close).iloc[-1]
    momentum = 0.0
    if hist.iloc[-1] > 0 and hist.iloc[-1] > hist.iloc[-2]:
        momentum += 50; reasons.append("MACD柱狀體轉強")
    elif hist.iloc[-1] > 0:
        momentum += 30
    if 55 <= rsi14 <= 80:                          # 強勢但未過熱的甜蜜帶
        momentum += 50; reasons.append(f"RSI={rsi14:.0f}強勢")
    elif rsi14 > 80:
        momentum += 15                             # 過熱扣分（避免追高最末段）
    momentum = min(momentum, 100.0)

    # --- (e) 三大法人籌碼 ---
    code = symbol.split(".")[0]
    chips_score = 50.0  # 無資料時給中性分
    net = chips.get(code)
    if net is not None:
        # 法人買超張數相對均量做標準化
        net_ratio = net / max(avg_vol20 / 1000.0, 1.0)
        chips_score = float(np.clip(50 + net_ratio * 200, 0, 100))
        if net > 0:
            reasons.append(f"法人買超{int(net)}張")

    # --- (f) 新聞 / 情緒 ---
    sent = sentiment.get(symbol, 0.0)
    sentiment_score = float(np.clip(50 + sent * 50, 0, 100))

    return StockScore(
        symbol=symbol, confidence=0.0, close=last_close,
        rs_score=rs_raw,            # 暫存原始 RS，稍後排名
        breakout_score=breakout, trend_score=trend,
        momentum_score=momentum, chips_score=chips_score,
        sentiment_score=sentiment_score, reasons=reasons,
    )


def _confidence(s: StockScore) -> float:
    """各子分數加權，回傳 0~100 信心分數。"""
    return (
        CFG.w_rs * s.rs_score
        + CFG.w_breakout * s.breakout_score
        + CFG.w_trend * s.trend_score
        + CFG.w_momentum * s.momentum_score
        + CFG.w_chips * s.chips_score
        + CFG.w_sentiment * s.sentiment_score
    )


def finalize_scores(scores: list[StockScore]) -> list[StockScore]:
    """把 RS 原始值轉成全市場百分位排名(0~100)，再加權算總信心分數。"""
    if not scores:
        return []

    rs_values = np.array([s.rs_score for s in scores])
    ranks = rs_values.argsort().argsort()  # 由小到大的名次
    rs_pct = ranks / max(len(scores) - 1, 1) * 100.0

    for s, pct in zip(scores, rs_pct):
        s.rs_score = float(pct)
        s.confidence = _confidence(s)
        if s.rs_score >= 80:
            s.reasons.insert(0, f"RS強度前{100 - int(s.rs_score)}%")

    scores.sort(key=lambda x: x.confidence, reverse=True)
    return scores


# =============================================================================
# 4. 每日選股主流程
# =============================================================================

def run_daily(universe: Optional[list[str]] = None) -> list[StockScore]:
    if universe is None:
        universe = build_universe()
    prices = fetch_prices(universe, CFG.lookback_days)
    index_close = fetch_index(CFG.lookback_days)
    chips = fetch_institutional_netbuy()

    # 第一階段：先用技術面 + 籌碼面評分（情緒暫設中性），求初步排名
    neutral_sent: dict[str, float] = {}
    raw_scores: list[StockScore] = []
    for sym, df in prices.items():
        s = compute_features(sym, df, index_close, chips, neutral_sent)
        if s is not None:
            raw_scores.append(s)

    scored = finalize_scores(raw_scores)

    # 第二階段：只對前段班 shortlist 抓新聞情緒（省流量），再修正信心分數
    if CFG.fetch_sentiment and scored:
        shortlist = scored[: CFG.sentiment_shortlist]
        sent = fetch_news_sentiment([s.symbol for s in shortlist])
        for s in shortlist:
            score = sent.get(s.symbol, 0.0)
            s.sentiment_score = float(np.clip(50 + score * 50, 0, 100))
            s.confidence = _confidence(s)
            if score > 0.2:
                s.reasons.append("新聞偏多")
            elif score < -0.2:
                s.reasons.append("新聞偏空")
        scored.sort(key=lambda x: x.confidence, reverse=True)

    # 寧缺勿濫第二道關卡：信心閥值 + 上限 20 檔
    qualified = [s for s in scored if s.confidence >= CFG.confidence_threshold]
    return qualified[: CFG.max_picks]


def print_report(picks: list[StockScore]) -> None:
    today = dt.date.today().strftime("%Y-%m-%d")
    print("\n" + "=" * 60)
    print(f"  台股動能飆股篩選報告  {today}")
    print(f"  信心閥值 = {CFG.confidence_threshold:.0f}  |  最多輸出 = {CFG.max_picks} 檔")
    print("=" * 60)

    if not picks:
        print("\n⚠️  今日無任何標的達到信心標準。")
        print("    （盤勢不佳，寧可空手 — 這是策略的一部分，不硬湊。）\n")
        return

    print(f"\n✅ 符合標準者共 {len(picks)} 檔（已按信心分數排序）：\n")
    header = f"{'排名':<4}{'代碼':<10}{'收盤':>8}{'信心':>7}   理由"
    print(header)
    print("-" * 60)
    for i, s in enumerate(picks, 1):
        reason = "、".join(s.reasons[:3]) if s.reasons else "-"
        print(f"{i:<4}{s.symbol:<10}{s.close:>8.2f}{s.confidence:>7.1f}   {reason}")

    print("\n" + "-" * 60)
    print(f"🏆 精選 Top {min(CFG.top_n, len(picks))}（最有機會噴出）：")
    for i, s in enumerate(picks[: CFG.top_n], 1):
        print(f"   {i}. {s.symbol}  信心 {s.confidence:.1f}")
        print(f"      RS={s.rs_score:.0f} 突破={s.breakout_score:.0f} "
              f"趨勢={s.trend_score:.0f} 動能={s.momentum_score:.0f} "
              f"籌碼={s.chips_score:.0f}")
    print("\n※ 本報告僅供研究教育用途，非投資建議。投資有風險。\n")


# =============================================================================
# 5. 回測框架（簡潔但具實戰意義）
# =============================================================================

def backtest(universe: list[str], test_days: int = 60) -> None:
    """
    走勢回放：對過去 test_days 個交易日，每天用「當天為止」的資料產生訊號，
    檢查訊號後 bt_hold_days 的報酬，統計勝率與平均報酬。
    避免未來函數：每個切片只用截至當日的價量。
    """
    print(f"\n開始回測（樣本期間：最近 {test_days} 交易日）...")
    prices = fetch_prices(universe, CFG.lookback_days + test_days)
    index_full = fetch_index(CFG.lookback_days + test_days)
    # 回測不重抓歷史法人/新聞，使用中性值（保守）
    chips: dict[str, float] = {}
    sentiment: dict[str, float] = {s: 0.0 for s in prices}

    wins = 0
    total = 0
    returns: list[float] = []

    # 找出共同的交易日索引
    sample_dates = None
    for df in prices.values():
        if len(df) > test_days + CFG.bt_hold_days:
            sample_dates = df.index
            break
    if sample_dates is None:
        print("資料不足，無法回測。")
        return

    test_idx = sample_dates[-(test_days + CFG.bt_hold_days):-CFG.bt_hold_days]

    for d in test_idx:
        day_scores: list[StockScore] = []
        for sym, df in prices.items():
            if d not in df.index:
                continue
            loc = df.index.get_loc(d)
            if loc < 120:
                continue
            hist = df.iloc[: loc + 1]
            idx_slice = index_full.loc[:d] if index_full is not None else None
            s = compute_features(sym, hist, idx_slice, chips, sentiment)
            if s is not None:
                day_scores.append(s)

        finalize_scores(day_scores)
        picks = [s for s in day_scores if s.confidence >= CFG.confidence_threshold][: CFG.top_n]

        # 評估每個訊號的前瞻報酬
        for s in picks:
            df = prices[s.symbol]
            loc = df.index.get_loc(d)
            if loc + CFG.bt_hold_days >= len(df):
                continue
            entry = df["Close"].iloc[loc]
            exit_ = df["Close"].iloc[loc + CFG.bt_hold_days]
            ret = float(exit_ / entry - 1.0)
            returns.append(ret)
            total += 1
            if ret >= CFG.bt_target_return:
                wins += 1

    print("\n" + "=" * 50)
    print("  回測結果")
    print("=" * 50)
    if total == 0:
        print("樣本期間內沒有產生任何訊號（門檻可能偏高）。")
        return
    win_rate = wins / total * 100
    avg_ret = np.mean(returns) * 100
    med_ret = np.median(returns) * 100
    print(f"訊號總數        : {total}")
    print(f"達標勝率(>+{CFG.bt_target_return*100:.0f}%) : {win_rate:.1f}%")
    print(f"平均{CFG.bt_hold_days}日報酬   : {avg_ret:+.2f}%")
    print(f"中位數報酬      : {med_ret:+.2f}%")
    print(f"最佳 / 最差     : {max(returns)*100:+.1f}% / {min(returns)*100:+.1f}%")
    print("=" * 50)
    print("※ 回測未計入交易成本與滑價，且未含歷史籌碼/新聞，結果偏保守。\n")


# =============================================================================
# 6. 進入點
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="台股動能飆股篩選器")
    parser.add_argument("--backtest", action="store_true", help="執行歷史回測")
    parser.add_argument("--days", type=int, default=60, help="回測天數")
    parser.add_argument("--threshold", type=float, help="覆寫信心閥值")
    parser.add_argument("--full", action="store_true",
                        help="掃描全上市櫃（預篩後取流動性最高的前 max_universe 檔）")
    parser.add_argument("--no-sentiment", action="store_true", help="關閉新聞情緒抓取")
    args = parser.parse_args()

    if args.threshold is not None:
        CFG.confidence_threshold = args.threshold
    if args.full:
        CFG.use_full_universe = True
    if args.no_sentiment:
        CFG.fetch_sentiment = False

    universe = build_universe()

    if args.backtest:
        backtest(universe, test_days=args.days)
    else:
        picks = run_daily(universe)
        print_report(picks)


if __name__ == "__main__":
    main()
