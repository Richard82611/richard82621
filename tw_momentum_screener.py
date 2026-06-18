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
import os
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

# 把 yfinance 對「查無資料/已下市」個股的警告靜音，避免畫面一堆紅字（程式本就會自動略過）
import logging
for _n in ("yfinance", "yfinance.utils", "yfinance.data", "peewee"):
    logging.getLogger(_n).setLevel(logging.CRITICAL)

# 所有輸出檔（CSV、圖）統一放在與本程式相同資料夾，並共用相同檔名前綴
_OUT_BASE = os.path.splitext(os.path.abspath(__file__))[0]  # 例：/path/tw_momentum_screener

try:
    import requests
except ImportError:
    requests = None


# =============================================================================
# 0. 參數設定（集中管理，方便手機上快速調整）
# =============================================================================

@dataclass
class Config:
    # --- 輸出控管（嚴格版預設）---
    max_picks: int = 10            # 每日最多輸出檔數
    top_n: int = 3                 # 首選推薦檔數（合格不足 3 檔時自動少給）
    confidence_threshold: float = 75.0  # 信心閥值；達不到就不輸出（寧缺勿濫）

    # --- 資料抓取 ---
    lookback_days: int = 260       # 抓約一年交易日，足夠算 52 週高點與 RS
    index_symbol: str = "^TWII"    # 加權指數，用來算相對強度 RS
    stale_max_days: int = 6        # 個股最後一根K落後「同儕最新交易日」超過此天數即剔除（停牌/下市）
    stale_abs_days: int = 12       # 落後「今日」超過此天數即剔除（含長假緩衝；防單檔池/整批皆停滯）

    # --- 技術門檻（硬性過濾，未過直接淘汰）---
    min_price: float = 10.0        # 排除雞蛋水餃股
    min_avg_volume: int = 1_000_000  # 20 日均量（股數）門檻，確保流動性
    volume_surge_mult: float = 1.5   # 當日量 > 1.5 倍 20 日均量 才算爆量
    near_high_pct: float = 0.90      # 收盤須站上 52 週高點的 90% 以上

    # --- 評分權重（總和 = 1.0）---
    # 設計理念見 README：飆股的本質是「相對強度 + 突破 + 量能」。
    # 「相對強度」拆成 個股 RS + 族群 RS（O'Neil/IBD 的產業群組排名概念）。
    w_rs: float = 0.15             # 個股相對強度 RS
    w_group: float = 0.10          # 族群（產業）相對強度 — 抓帶頭族群
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

    # --- 族群分析 ---
    use_industry: bool = True      # 是否抓產業別並計算族群相對強度

    # --- FinMind 歷史法人（讓回測納入籌碼面）---
    use_finmind: bool = False      # True = 回測時用 FinMind 抓歷史三大法人
    finmind_token: str = ""        # FinMind API token（選填，留空用免費額度；亦讀環境變數 FINMIND_TOKEN）

    # --- 回測 ---
    bt_hold_days: int = 5          # 訊號後持有天數
    bt_target_return: float = 0.03 # 視為「成功」的報酬門檻（5 日 +3%）


CFG = Config()


# 預設股票池：台灣 50 + 中型 100 常見成分與熱門題材股，並涵蓋一批上櫃熱門股。
# 想完整掃描「全上市櫃」請用 --full（會自動抓 TWSE+TPEx 全市場再預篩）。
# 可自行增減；代碼後綴 .TW（上市）或 .TWO（上櫃）。
DEFAULT_UNIVERSE = [
    # --- 上市 .TW ---
    "2330.TW", "2317.TW", "2454.TW", "2308.TW", "2382.TW", "2303.TW",
    "3711.TW", "2412.TW", "2881.TW", "2882.TW", "2891.TW", "1303.TW",
    "1301.TW", "2002.TW", "2603.TW", "2609.TW", "2615.TW", "3008.TW",
    "3034.TW", "3037.TW", "3231.TW", "2376.TW", "2377.TW", "2379.TW",
    "3661.TW", "3443.TW", "4966.TW", "5269.TW", "6669.TW", "8046.TW",
    "2357.TW", "2356.TW", "2049.TW", "1519.TW", "1513.TW", "6505.TW",
    "2207.TW", "9910.TW", "2105.TW", "1216.TW", "2912.TW", "2884.TW",
    "3017.TW", "3035.TW", "2345.TW", "3045.TW", "4938.TW", "2408.TW",
    # --- 上櫃 .TWO ---
    "5483.TWO", "6182.TWO", "8069.TWO", "5274.TWO", "3260.TWO", "6488.TWO",
    "6510.TWO", "6147.TWO", "5347.TWO", "3293.TWO", "8086.TWO", "1565.TWO",
    "6531.TWO", "6533.TWO", "8299.TWO", "3338.TWO", "5439.TWO", "4128.TWO",
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
    # loss=0（期間全漲）時 gain/loss=inf → RSI=100（正確的強勢表現）；
    # gain=loss=0（無波動）→ NaN，補中性 50。
    rs = gain / loss
    return (100 - (100 / (1 + rs))).fillna(50.0)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """平均真實波幅 (ATR)，用來設動態停損停利。"""
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


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
            # 只留 1101~9999 的普通股；排除 00 開頭的 ETF/ETN（如 0050、0056）
            if not (code.isdigit() and len(code) == 4 and not code.startswith("0")):
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
            # 排除 00 開頭的 ETF/ETN，只留普通股
            if not (code.isdigit() and len(code) == 4 and not code.startswith("0")):
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

    twse_rows = fetch_twse_all_day()
    tpex_rows = fetch_tpex_all_day()
    # 任一市場抓取失敗就明確告警，避免「宣稱掃全上市櫃、實際只掃單一市場」
    if not twse_rows:
        print("⚠️ 上市清單抓取失敗，本次僅涵蓋上櫃。")
    if not tpex_rows:
        print("⚠️ 上櫃清單抓取失敗，本次僅涵蓋上市。")
    allrows = twse_rows + tpex_rows
    if not allrows:
        print("全市場清單皆不可用，降級為內建股票池（含上市櫃）。")
        return DEFAULT_UNIVERSE

    # 預篩：股價門檻 + 寬鬆的今日量門檻（僅為縮小下載量，故用 min_avg_volume 的一半，
    # 避免把「20日均量足夠、只是今天較清淡」的個股直接擋掉；真正的流動性硬門檻
    # 仍由 compute_features 以「前 20 日均量」把關）。
    vol_floor = CFG.min_avg_volume * 0.5
    cands = [r for r in allrows
             if r["close"] >= CFG.min_price and r["volume"] >= vol_floor]
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
            if raw is not None and not raw.empty:
                if isinstance(raw.columns, pd.MultiIndex) and sym in raw.columns.levels[0]:
                    df = raw[sym].dropna()
                elif len(symbols) == 1:
                    # 單一代碼時 yfinance 不回傳 MultiIndex，直接用 raw 避免重複下載
                    df = raw.dropna()
                else:
                    df = yf.download(sym, start=start, end=end, auto_adjust=True, progress=False)
            else:
                df = yf.download(sym, start=start, end=end, auto_adjust=True, progress=False)
            if df is None or len(df) < 60:
                continue
            # yfinance >= 0.2.40 預設 multi_level_index=True，單檔/逐檔下載可能殘留
            # 多層欄位 ('Close','2330.TW')，先壓平成單層，避免後續取到 DataFrame 而非 Series
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.rename(columns=str.title)[["Open", "High", "Low", "Close", "Volume"]]
            out[sym] = df.dropna()
        except Exception:
            continue

    # 剔除「資料停滯」的個股，避免拿舊資料當今日突破標的。雙重判準：
    #  (a) 相對整批最新交易日落後 > stale_max_days（抓出在新鮮批次中被停牌者）；
    #  (b) 相對「今天」落後 > stale_abs_days（單檔池/整批皆停滯時，批次最大值本身就舊，
    #      故另用今日為獨立基準；門檻較寬以容忍農曆年等長假）。
    if out:
        today = pd.Timestamp(dt.date.today())
        batch_latest = max(df.index[-1] for df in out.values())
        stale = [sym for sym, df in out.items()
                 if (batch_latest - df.index[-1]).days > CFG.stale_max_days
                 or (today - df.index[-1]).days > CFG.stale_abs_days]
        for sym in stale:
            del out[sym]
        if stale:
            print(f"剔除 {len(stale)} 檔資料停滯（落後同儕 >{CFG.stale_max_days} 天"
                  f"或落後今日 >{CFG.stale_abs_days} 天）。")
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
    抓三大法人「個股」買賣超（上市 TWSE + 上櫃 TPEx，免金鑰）。
    回傳 {股號(不含後綴): 買賣超張數}。任何失敗都回傳 {} 並降級。

    會往前找最多 7 天，解決「週末/假日/盤中(15:30 前)當日尚無資料」的情況。
    """
    if requests is None:
        return {}
    url = "https://www.twse.com.tw/rwd/zh/fund/T86"
    result: dict[str, float] = {}
    twse_ok = False

    # 往前找最近一個有資料的交易日（最多回溯 7 天）
    for i in range(7):
        date_str = (dt.date.today() - dt.timedelta(days=i)).strftime("%Y%m%d")
        params = {"date": date_str, "selectType": "ALL", "response": "json"}
        try:
            r = requests.get(url, params=params, timeout=10)
            data = r.json()
            if data.get("stat") != "OK":
                continue  # 該日非交易日或尚無資料，往前一天
            for row in data.get("data", []):
                code = row[0].strip()
                net = _to_float(row[-1])  # 最後一欄為三大法人買賣超股數
                if net is not None:
                    result[code] = net / 1000.0  # 股 -> 張
            print(f"取得 {date_str} 上市三大法人買賣超 {len(result)} 檔。")
            twse_ok = True
            break
        except Exception:
            continue

    # 併入上櫃法人資料（櫃買 OpenAPI 自動回傳最近交易日）
    tpex = _fetch_tpex_netbuy()
    result.update(tpex)

    # 分市場回報：任一邊不可用就明確告警，避免「宣稱含上市櫃、其實某一市場全中性」
    if not twse_ok:
        print("⚠️ 上市法人資料不可用，上市個股籌碼以中性計（不影響上櫃）。")
    if not tpex:
        print("⚠️ 上櫃法人資料不可用，上櫃個股籌碼以中性計（不影響上市）。")
    if not result:
        print("無法取得任何法人資料（全部降級為純價量）。")
    else:
        print(f"法人買賣超合計 {len(result)} 檔（上市 {'OK' if twse_ok else '缺'} / "
              f"上櫃 {'OK' if tpex else '缺'}）。")
    return result


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
            # 欄名各版本略有差異，盡量找「三大法人買賣超」總計欄。
            # 櫃買 tpex_3insti_daily_trading 的總計欄位實際為 TotalDifference。
            net = None
            for k in ("TotalDifference", "TotalBuySellShares",
                      "TotalInstitutionalInvestorsNetBuySell",
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


# MOPS 公開資訊觀測站「產業別」代碼表（上市與上櫃共用），用來把 TPEx 的數字代碼
# 正規化成與 TWSE 一致的中文產業名，避免同產業跨市場被拆成兩個標籤。
_INDUSTRY_CODE_MAP = {
    "01": "水泥工業", "02": "食品工業", "03": "塑膠工業", "04": "紡織纖維",
    "05": "電機機械", "06": "電器電纜", "08": "玻璃陶瓷", "09": "造紙工業",
    "10": "鋼鐵工業", "11": "橡膠工業", "12": "汽車工業", "14": "建材營造",
    "15": "航運業", "16": "觀光餐旅", "17": "金融保險業", "18": "貿易百貨",
    "20": "其他業", "21": "化學工業", "22": "生技醫療業", "23": "油電燃氣業",
    "24": "半導體業", "25": "電腦及週邊設備業", "26": "光電業", "27": "通信網路業",
    "28": "電子零組件業", "29": "電子通路業", "30": "資訊服務業", "31": "其他電子業",
    "32": "文化創意業", "33": "農業科技業", "34": "電子商務", "35": "綠能環保",
    "36": "數位雲端", "37": "運動休閒", "38": "居家生活",
}


def _normalize_industry(value: str) -> str:
    """把產業別正規化為名稱：純數字代碼查 MOPS 表（未知則維持原值），名稱原樣保留。"""
    v = str(value).strip()
    if v.isdigit():
        return _INDUSTRY_CODE_MAP.get(v.zfill(2), v)
    return v


def _clean_name(nm: str) -> str:
    """把公司全名整理成簡短名稱：去掉法律後綴、括號註記，過長再截斷，避免撐爆報告排版。"""
    nm = str(nm).strip()
    for suf in ("股份有限公司", "股份有限", "控股股份有限公司", "控股公司"):
        if nm.endswith(suf):
            nm = nm[: -len(suf)]
            break
    nm = nm.strip("　 ()（）")
    return nm[:8]  # 保留 -KY 等有意義字尾，但整體不超過 8 字


def fetch_company_meta() -> tuple[dict, dict]:
    """
    抓個股 -> (產業別, 公司簡稱) 對照（TWSE + TPEx 公司基本資料 OpenAPI，免金鑰）。
    回傳 (industry_map, name_map)，皆為 {股號: 值}。任何失敗回 ({}, {}) 並降級。
    TWSE 提供產業「名稱」、TPEx 多為產業「代碼」，故統一以 MOPS 代碼表正規化成名稱，
    避免同一產業在上市/上櫃被拆成兩個標籤。
    """
    if requests is None:
        return {}, {}
    industry: dict[str, str] = {}
    names: dict[str, str] = {}
    sources = [
        ("https://openapi.twse.com.tw/v1/opendata/t187ap03_L",
         ("公司代號", "Code"), ("產業別", "Industry"),
         ("公司簡稱", "公司名稱", "CompanyName", "Name")),
        ("https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O",
         ("SecuritiesCompanyCode", "公司代號"),
         ("產業別", "Industry", "SecuritiesIndustry", "SecuritiesIndustryCode"),
         ("公司簡稱", "CompanyName", "公司名稱", "CompanyAbbreviation", "Name")),
    ]
    for url, code_keys, ind_keys, name_keys in sources:
        try:
            rows = requests.get(url, timeout=15).json()
            for row in rows:
                code = next((str(row[k]).strip() for k in code_keys if k in row and row[k]), "")
                if not (code.isdigit() and len(code) == 4):
                    continue
                ind = _normalize_industry(
                    next((str(row[k]).strip() for k in ind_keys if k in row and row[k]), ""))
                nm = next((str(row[k]).strip() for k in name_keys if k in row and row[k]), "")
                if ind:
                    industry[code] = ind
                if nm:
                    names[code] = _clean_name(nm)
        except Exception:
            continue
    if industry or names:
        print(f"取得公司基本資料：產業 {len(industry)} 檔、名稱 {len(names)} 檔。")
    else:
        print("公司基本資料不可用（族群以個股 RS 替代、名稱以代碼顯示）。")
    return industry, names


def fetch_industry_map() -> dict[str, str]:
    """（相容用）只取產業別對照。"""
    return fetch_company_meta()[0]


def fetch_finmind_chips(symbols: list[str], start_date: str,
                        end_date: Optional[str] = None) -> dict[str, dict[str, float]]:
    """
    用 FinMind 抓「歷史」三大法人買賣超，供回測納入籌碼面。
    回傳 {股號: {日期(YYYY-MM-DD): 買賣超張數}}；失敗的個股略過。
    免金鑰可用（有額度限制）；設 token 可提高額度（CFG.finmind_token 或環境變數 FINMIND_TOKEN）。
    """
    if requests is None:
        return {}
    token = CFG.finmind_token or os.environ.get("FINMIND_TOKEN", "")
    end_date = end_date or dt.date.today().strftime("%Y-%m-%d")
    api = "https://api.finmindtrade.com/api/v4/data"
    out: dict[str, dict[str, float]] = {}

    codes = [s.split(".")[0] for s in symbols]
    print(f"FinMind 抓取 {len(codes)} 檔歷史法人（{start_date} ~ {end_date}）...")
    for code in codes:
        params = {
            "dataset": "TaiwanStockInstitutionalInvestorsBuySell",
            "data_id": code, "start_date": start_date, "end_date": end_date,
        }
        if token:
            params["token"] = token
        try:
            resp = requests.get(api, params=params, timeout=15).json()
            if resp.get("status") != 200 or not resp.get("data"):
                continue
            df = pd.DataFrame(resp["data"])
            # 各法人別 buy/sell 加總後 net = (buy - sell)，再依日期彙總，股 -> 張
            df["net"] = (df["buy"] - df["sell"]) / 1000.0
            by_date = df.groupby("date")["net"].sum()
            out[code] = {d: float(v) for d, v in by_date.items()}
        except Exception:
            continue
        time.sleep(0.15)  # 輕微節流，避免觸發免費額度限制
    print(f"FinMind 取得 {len(out)} 檔歷史法人資料。")
    return out


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

def market_of(symbol: str) -> str:
    """依代碼後綴判斷市場別。"""
    return "上櫃" if symbol.upper().endswith(".TWO") else "上市"


@dataclass
class StockScore:
    symbol: str
    confidence: float
    close: float
    rs_score: float = 0.0
    group_score: float = 0.0    # 族群（產業）相對強度百分位
    breakout_score: float = 0.0
    trend_score: float = 0.0
    momentum_score: float = 0.0
    chips_score: float = 0.0
    sentiment_score: float = 0.0
    name: str = ""              # 公司簡稱
    industry: str = ""          # 產業別
    is_group_leader: bool = False  # 是否為所屬族群的帶頭股
    # 風控建議（ATR 動態停損停利）
    atr: float = 0.0
    stop_loss: float = 0.0      # 建議停損價
    take_profit: float = 0.0    # 建議停利價
    reasons: list[str] = field(default_factory=list)

    @property
    def market(self) -> str:
        return market_of(self.symbol)


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
    # 量能基準用「前 20 日」（排除當日），避免今日爆量灌大分母、低估爆量倍數
    avg_vol20 = float(vol.iloc[-21:-1].mean()) if len(vol) >= 21 else float(vol.tail(20).mean())
    if last_close < CFG.min_price:
        return None
    if avg_vol20 < CFG.min_avg_volume:
        return None

    # 52 週高點（約 252 日），用「前一日為止」（排除今日）作為突破基準：
    # 今日盤中創高若也算進 high_52w，會讓「收盤站上前高」被誤判為僅逼近而非突破。
    high_52w = float(high.iloc[:-1].tail(252).max()) if len(high) >= 2 else float(high.iloc[-1])
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

    # --- 風控：ATR 動態停損停利（停損 2 ATR，停利 3 ATR，風報比 1:1.5）---
    atr14 = atr(df).iloc[-1]
    atr14 = float(atr14) if pd.notna(atr14) else 0.0
    stop_loss = round(last_close - 2 * atr14, 2) if atr14 > 0 else 0.0
    take_profit = round(last_close + 3 * atr14, 2) if atr14 > 0 else 0.0

    return StockScore(
        symbol=symbol, confidence=0.0, close=last_close,
        rs_score=rs_raw,            # 暫存原始 RS，稍後排名
        breakout_score=breakout, trend_score=trend,
        momentum_score=momentum, chips_score=chips_score,
        sentiment_score=sentiment_score,
        atr=round(atr14, 2), stop_loss=stop_loss, take_profit=take_profit,
        reasons=reasons,
    )


def _confidence(s: StockScore) -> float:
    """各子分數加權，回傳 0~100 信心分數。"""
    return (
        CFG.w_rs * s.rs_score
        + CFG.w_group * s.group_score
        + CFG.w_breakout * s.breakout_score
        + CFG.w_trend * s.trend_score
        + CFG.w_momentum * s.momentum_score
        + CFG.w_chips * s.chips_score
        + CFG.w_sentiment * s.sentiment_score
    )


def _assign_group_scores(scores: list[StockScore]) -> None:
    """
    依產業別計算「族群相對強度」：
    - 每個族群的強度 = 該族群成員個股 RS 百分位的平均。
    - 個股 group_score = 其所屬族群強度（族群越強，整體分數越高）。
    - 各族群中 RS 最高者標記為「族群帶頭股」。
    無產業資料的個股，group_score 退化為自身 RS（中性，不額外加減分）。
    """
    by_ind: dict[str, list[StockScore]] = {}
    for s in scores:
        if s.industry:
            by_ind.setdefault(s.industry, []).append(s)

    group_strength = {ind: float(np.mean([m.rs_score for m in members]))
                      for ind, members in by_ind.items()}

    for s in scores:
        if s.industry and s.industry in group_strength:
            s.group_score = group_strength[s.industry]
        else:
            s.group_score = s.rs_score  # 無族群資料 → 中性

    # 標記每個族群的帶頭股（成員 >= 2 才有意義）
    for ind, members in by_ind.items():
        if len(members) < 2:
            continue
        leader = max(members, key=lambda m: m.rs_score)
        if leader.rs_score >= 70 and group_strength[ind] >= 60:
            leader.is_group_leader = True


def build_group_universe(prices: dict, index_close, industry: dict,
                         names: Optional[dict] = None) -> list[StockScore]:
    """
    用「全股票池」（不套用價量/創高的硬性過濾）計算各個股 RS 百分位與族群強度，
    供族群輪動排行與圖表使用 —— 確保族群強度反映整個產業，而非僅突破候選。
    回傳已含 rs_score(百分位) / group_score / is_group_leader / industry / name 的清單。
    """
    names = names or {}
    items: list[StockScore] = []
    for sym, df in prices.items():
        if df is None or len(df) < 126:  # RS 需足夠歷史
            continue
        close = df["Close"]
        code = sym.split(".")[0]
        s = StockScore(symbol=sym, confidence=0.0, close=float(close.iloc[-1]))
        s.rs_score = _rel_strength_pct(close, index_close)   # 暫存原始 RS
        s.industry = industry.get(code, "")
        s.name = names.get(code, "")
        items.append(s)
    if not items:
        return []
    rs_pct = _percentile_rank([s.rs_score for s in items])
    for s, pct in zip(items, rs_pct):
        s.rs_score = float(pct)
    _assign_group_scores(items)
    return items


def _group_strength_leaders(group_items: list[StockScore]):
    """從全市場族群清單擷取 {產業: 族群強度} 與 {產業: 帶頭股代碼}。"""
    by_ind: dict[str, list[StockScore]] = {}
    for s in group_items:
        if s.industry:
            by_ind.setdefault(s.industry, []).append(s)
    strength = {ind: float(np.mean([m.group_score for m in members]))
                for ind, members in by_ind.items()}
    leaders = {s.industry: s.symbol for s in group_items if s.is_group_leader}
    return strength, leaders


def _universe_context(group_items: list[StockScore]):
    """由全市場族群清單導出 (個股RS百分位 map, 族群強度, 帶頭股)，供評分採用全市場排名。"""
    rs_pct_map = {s.symbol: s.rs_score for s in group_items}
    strength, leaders = _group_strength_leaders(group_items)
    return rs_pct_map, strength, leaders


def _percentile_rank(values) -> np.ndarray:
    """把一組數值轉成 0~100 百分位。相同數值給相同百分位（average ties），
    避免僅憑輸入順序就讓相同 RS 的個股分到不同名次。"""
    s = pd.Series(values, dtype="float64")
    if len(s) <= 1:
        return np.full(len(s), 50.0)
    ranks = s.rank(method="average")          # 平手取平均名次（1..n）
    return ((ranks - 1) / (len(s) - 1) * 100.0).to_numpy()


def finalize_scores(scores: list[StockScore],
                    rs_pct_map: Optional[dict] = None,
                    group_strength: Optional[dict] = None,
                    group_leaders: Optional[dict] = None) -> list[StockScore]:
    """把 RS 原始值轉成百分位排名(0~100)，計算族群強度，再加權算總信心。

    rs_pct_map / group_strength / group_leaders 若提供（皆來自「全股票池」而非僅
    突破候選），則個股 RS 與族群分數都採全市場排名，避免硬性過濾後因存活檔數少
    而扭曲分數；否則退化為以傳入名單自行計算（回測舊路徑 / 單元測試用）。
    """
    if not scores:
        return []

    if rs_pct_map is not None:
        # 用全市場 RS 百分位（找不到者給中性 50）
        for s in scores:
            s.rs_score = float(rs_pct_map.get(s.symbol, 50.0))
    else:
        rs_pct = _percentile_rank([s.rs_score for s in scores])
        for s, pct in zip(scores, rs_pct):
            s.rs_score = float(pct)

    # 族群相對強度：優先用全市場族群強度，否則以傳入名單估算
    if group_strength is not None:
        leaders = group_leaders or {}
        for s in scores:
            s.group_score = group_strength.get(s.industry, s.rs_score)
            s.is_group_leader = bool(s.industry) and leaders.get(s.industry) == s.symbol
    else:
        _assign_group_scores(scores)

    for s in scores:
        s.confidence = _confidence(s)
        if s.rs_score >= 80:
            s.reasons.insert(0, f"RS強度前{max(1, 100 - int(s.rs_score))}%")
        if s.is_group_leader:
            s.reasons.insert(0, f"族群帶頭股({s.industry})")

    scores.sort(key=lambda x: x.confidence, reverse=True)
    return scores


# =============================================================================
# 4. 每日選股主流程
# =============================================================================

def _score_universe(universe: list[str]):
    """抓資料並對整個股票池評分（含族群強度），回傳尚未套用新聞情緒的完整排名。
    回傳 (scored, prices, index_close, industry, group_items)，供選股與族群分析共用。
    族群強度以「全股票池」計算（不受個股硬性過濾影響）。"""
    prices = fetch_prices(universe, CFG.lookback_days)
    index_close = fetch_index(CFG.lookback_days)
    chips = fetch_institutional_netbuy()
    industry, names = fetch_company_meta() if CFG.use_industry else ({}, {})

    # 全市場排名脈絡（不套用突破/流動性過濾）：個股 RS 百分位 + 族群強度 + 帶頭股。
    # 即使沒有產業資料（--no-industry 或來源不可用），仍計算全市場 RS 百分位，
    # 避免個股 RS 因硬性過濾後存活檔數多寡而失真（族群分數則自動退化為中性）。
    group_items = build_group_universe(prices, index_close, industry, names)
    rs_map, gstrength, gleaders = _universe_context(group_items)

    neutral_sent: dict[str, float] = {}
    raw_scores: list[StockScore] = []
    for sym, df in prices.items():
        s = compute_features(sym, df, index_close, chips, neutral_sent)
        if s is not None:
            code = sym.split(".")[0]
            s.industry = industry.get(code, "")
            s.name = names.get(code, "")
            raw_scores.append(s)

    scored = finalize_scores(raw_scores, rs_map or None, gstrength or None, gleaders or None)
    return scored, prices, index_close, industry, group_items


def run_daily(universe: Optional[list[str]] = None,
              show_groups: bool = False) -> list[StockScore]:
    if universe is None:
        universe = build_universe()

    # 第一階段：技術面 + 籌碼面 + 族群強度評分（情緒暫設中性），求初步排名
    scored, _prices, _idx, _ind, group_items = _score_universe(universe)

    # 每日流程：先呈現族群輪動脈絡（以全市場計算，強勢族群帶動帶頭股），再看選股
    if show_groups and CFG.use_industry:
        print_group_ranking(group_items)

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

    n_listed = sum(1 for s in picks if s.market == "上市")
    n_otc = len(picks) - n_listed
    print(f"\n✅ 符合標準者共 {len(picks)} 檔（上市 {n_listed} / 上櫃 {n_otc}，按信心排序）：\n")
    for i, s in enumerate(picks, 1):
        reason = "、".join(s.reasons[:3]) if s.reasons else "-"
        nm = s.name or "-"
        print(f"{i:<3}{s.symbol:<9}{nm:<7}{s.market:<4}收{s.close:>8.2f}  信心{s.confidence:>5.1f}")
        print(f"     {reason}")

    print("\n" + "-" * 64)
    print(f"🏆 精選 Top {min(CFG.top_n, len(picks))}（最有機會噴出，含風控建議）：")
    for i, s in enumerate(picks[: CFG.top_n], 1):
        tag = " 👑族群帶頭" if s.is_group_leader else ""
        ind = f"｜{s.industry}" if s.industry else ""
        nm = f" {s.name}" if s.name else ""
        print(f"   {i}. {s.symbol}{nm}（{s.market}{ind}）  信心 {s.confidence:.1f}{tag}")
        print(f"      RS={s.rs_score:.0f} 族群={s.group_score:.0f} "
              f"突破={s.breakout_score:.0f} 趨勢={s.trend_score:.0f} "
              f"動能={s.momentum_score:.0f} 籌碼={s.chips_score:.0f}")
        if s.atr > 0:
            risk = s.close - s.stop_loss
            reward = s.take_profit - s.close
            rr = reward / risk if risk > 0 else 0
            print(f"      風控：收盤 {s.close:.2f}｜停損 {s.stop_loss:.2f}"
                  f"（-{risk:.2f}）｜停利 {s.take_profit:.2f}"
                  f"（+{reward:.2f}）｜風報比 1:{rr:.1f}")
    print("\n※ 本報告僅供研究教育用途，非投資建議。投資有風險。\n")


def export_csv(picks: list[StockScore], path: Optional[str] = None) -> Optional[str]:
    """把選股結果存成 CSV（預設與程式同資料夾、同檔名前綴）。回傳檔案路徑；無資料回 None。"""
    if path is None:
        path = f"{_OUT_BASE}_picks.csv"
    if not picks:
        print("無標的可匯出。")
        return None
    rows = [{
        "日期": dt.date.today().strftime("%Y-%m-%d"),
        "代碼": s.symbol, "名稱": s.name, "市場": s.market, "產業": s.industry,
        "族群帶頭": "是" if s.is_group_leader else "",
        "收盤": s.close,
        "信心分數": round(s.confidence, 1), "RS": round(s.rs_score),
        "族群": round(s.group_score), "突破": round(s.breakout_score),
        "趨勢": round(s.trend_score), "動能": round(s.momentum_score),
        "籌碼": round(s.chips_score), "情緒": round(s.sentiment_score),
        "停損": s.stop_loss, "停利": s.take_profit, "ATR": s.atr,
        "理由": "、".join(s.reasons),
    } for s in picks]
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, encoding="utf-8-sig")  # utf-8-sig 讓 Excel 不亂碼
    print(f"已匯出 {len(picks)} 檔到 {path}")
    return path


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
    sentiment: dict[str, float] = {s: 0.0 for s in prices}  # 回測不重抓新聞，用中性值
    industry = fetch_industry_map() if CFG.use_industry else {}

    # FinMind 歷史三大法人：讓回測也納入籌碼面（否則 chips_score 用中性 50）
    chips_hist: dict[str, dict[str, float]] = {}
    if CFG.use_finmind and prices:
        bt_start = (dt.date.today() - dt.timedelta(
            days=int((CFG.lookback_days + test_days) * 1.6))).strftime("%Y-%m-%d")
        chips_hist = fetch_finmind_chips(list(prices.keys()), bt_start)

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
        d_str = d.strftime("%Y-%m-%d")
        # 當日的歷史法人籌碼（FinMind）；無資料則為空 dict → 籌碼分數中性
        chips_d = {code: by_date[d_str]
                   for code, by_date in chips_hist.items() if d_str in by_date}
        idx_slice = index_full.loc[:d] if index_full is not None else None

        # 當日「截至 d」的價格切片，供全市場族群/RS 排名（與 run_daily 一致，
        # 無產業資料時仍提供全市場 RS 百分位）
        prices_asof = {sym: df.loc[:d] for sym, df in prices.items() if d in df.index}
        group_items = build_group_universe(prices_asof, idx_slice, industry)
        rs_map, gstrength, gleaders = _universe_context(group_items)

        day_scores: list[StockScore] = []
        for sym, df in prices.items():
            if d not in df.index:
                continue
            loc = df.index.get_loc(d)
            if loc < 120:
                continue
            hist = df.iloc[: loc + 1]
            s = compute_features(sym, hist, idx_slice, chips_d, sentiment)
            if s is not None:
                s.industry = industry.get(sym.split(".")[0], "")
                day_scores.append(s)

        finalize_scores(day_scores, rs_map or None, gstrength or None, gleaders or None)
        picks = [s for s in day_scores if s.confidence >= CFG.confidence_threshold][: CFG.top_n]

        # 評估每個訊號的前瞻報酬。訊號在收盤後產生（含盤後籌碼），
        # 故以「隔日開盤價」買入才符合實際交易，避免未來函數(look-ahead bias)。
        # 持有 bt_hold_days 個交易日：買在 T+1 開盤，賣在 T+bt_hold_days 收盤。
        for s in picks:
            df = prices[s.symbol]
            loc = df.index.get_loc(d)
            if loc + CFG.bt_hold_days >= len(df):
                continue
            entry = df["Open"].iloc[loc + 1]
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
    chips_note = "已納入 FinMind 歷史法人籌碼" if (CFG.use_finmind and chips_hist) else "未含歷史籌碼"
    print(f"※ 回測未計入交易成本與滑價；籌碼面：{chips_note}；未含歷史新聞。\n")


# =============================================================================
# 5.5 族群輪動分析與視覺化
# =============================================================================

def rank_groups(scores: list[StockScore]) -> list[dict]:
    """
    依族群強度排序產業。回傳 [{產業, 強度, 檔數, 帶頭股, 帶頭股RS}]，由強到弱。
    需傳入已 finalize_scores 的完整名單（含 rs_score / group_score / industry）。
    """
    by_ind: dict[str, list[StockScore]] = {}
    for s in scores:
        if s.industry:
            by_ind.setdefault(s.industry, []).append(s)

    rows = []
    for ind, members in by_ind.items():
        strength = float(np.mean([m.group_score for m in members]))
        leader = max(members, key=lambda m: m.rs_score)
        leader_label = f"{leader.symbol} {leader.name}".strip()
        rows.append({
            "industry": ind, "strength": strength, "count": len(members),
            "leader": leader_label, "leader_rs": leader.rs_score,
        })
    rows.sort(key=lambda r: r["strength"], reverse=True)
    return rows


def print_group_ranking(scores: list[StockScore], top_k: int = 15) -> None:
    """純文字族群強度排行榜（零相依，手機必看）。"""
    rows = rank_groups(scores)
    print("\n" + "=" * 60)
    print(f"  族群強度排行榜  {dt.date.today():%Y-%m-%d}（強度=族群成員 RS 百分位均值）")
    print("=" * 60)
    if not rows:
        print("無產業別資料（可能離線或來源暫時不可用），無法產生族群排行。\n")
        return
    print(f"{'排名':<4}{'產業':<14}{'強度':>6}{'檔數':>5}   帶頭股")
    print("-" * 60)
    for i, r in enumerate(rows[:top_k], 1):
        bar = "█" * int(r["strength"] / 5)  # 簡易強度長條（0~100 -> 0~20 格）
        print(f"{i:<4}{r['industry']:<14}{r['strength']:>6.0f}{r['count']:>5}   "
              f"{r['leader']}(RS{r['leader_rs']:.0f})  {bar}")
    print("\n💡 資金多集中在前段族群；族群越強，其帶頭股越值得優先觀察。\n")


def compute_group_rotation(prices: dict, index_close, industry: dict,
                           lookback_days: int = 20) -> "pd.DataFrame":
    """
    計算近 lookback_days 個交易日、各產業的「族群相對強度」時間序列。
    每天對所有個股算趨勢 RS → 當日橫斷面百分位排名 → 依產業取平均。
    回傳 DataFrame（index=日期、columns=產業、值=0~100 族群強度）。
    """
    # 取最長的價格序列當作交易日基準
    base_dates = None
    for df in prices.values():
        if base_dates is None or len(df) > len(base_dates):
            base_dates = df.index
    if base_dates is None:
        return pd.DataFrame()
    days = list(base_dates[-lookback_days:])

    records: dict = {}
    for d in days:
        rs_today: dict[str, float] = {}
        for sym, df in prices.items():
            if d not in df.index:
                continue
            loc = df.index.get_loc(d)
            if loc < 126:  # 至少半年資料才算 RS
                continue
            close = df["Close"].iloc[: loc + 1]
            idx_slice = index_close.loc[:d] if index_close is not None else None
            rs_today[sym] = _rel_strength_pct(close, idx_slice)
        if len(rs_today) < 2:
            continue
        # 橫斷面百分位排名（平手取平均名次）
        syms = list(rs_today)
        pct = _percentile_rank([rs_today[s] for s in syms])
        # 依產業彙總
        ind_acc: dict[str, list[float]] = {}
        for s, p in zip(syms, pct):
            ind = industry.get(s.split(".")[0], "")
            if ind:
                ind_acc.setdefault(ind, []).append(p)
        records[d] = {ind: float(np.mean(v)) for ind, v in ind_acc.items()}

    return pd.DataFrame.from_dict(records, orient="index").sort_index()


def plot_group_strength(scores: list[StockScore], path: Optional[str] = None) -> Optional[str]:
    """族群強度橫條圖（快照）。需 matplotlib；未安裝則略過。"""
    if path is None:
        path = f"{_OUT_BASE}_group_strength.png"
    try:
        import matplotlib
        matplotlib.use("Agg")  # 無頭環境也能存檔
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安裝 matplotlib，略過繪圖（!pip install matplotlib）。")
        return None
    _set_cjk_font()

    rows = rank_groups(scores)
    if not rows:
        print("無族群資料可繪圖。")
        return None
    rows = rows[:12][::-1]  # 取前 12 強，橫條由下而上
    labels = [r["industry"] for r in rows]
    values = [r["strength"] for r in rows]
    colors = ["#d62728" if v >= 70 else "#ff7f0e" if v >= 50 else "#7f7f7f" for v in values]

    plt.figure(figsize=(8, 6))
    plt.barh(labels, values, color=colors)
    plt.xlabel("族群強度（0~100）")
    plt.title(f"台股族群強度排行 {dt.date.today():%Y-%m-%d}")
    plt.xlim(0, 100)
    plt.tight_layout()
    _save_fig(plt, path)
    return path


def plot_group_rotation(rotation: "pd.DataFrame", top_k: int = 6,
                        path: Optional[str] = None) -> Optional[str]:
    """族群輪動趨勢線：近 N 日各族群強度變化，看誰在升溫/退燒。"""
    if path is None:
        path = f"{_OUT_BASE}_group_rotation.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安裝 matplotlib，略過繪圖（!pip install matplotlib）。")
        return None
    _set_cjk_font()

    if rotation is None or rotation.empty:
        print("輪動資料不足，無法繪圖。")
        return None
    # 取最新一天最強的前 top_k 族群
    latest = rotation.iloc[-1].dropna().sort_values(ascending=False)
    cols = list(latest.index[:top_k])

    plt.figure(figsize=(9, 5))
    for c in cols:
        plt.plot(rotation.index, rotation[c], marker="o", markersize=3, label=c)
    plt.ylabel("族群強度（0~100）")
    plt.title(f"台股族群輪動（近 {len(rotation)} 交易日）")
    plt.ylim(0, 100)
    plt.legend(loc="best", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    _save_fig(plt, path)
    return path


def _set_cjk_font() -> None:
    """繪圖前設定中文字型，避免中文變方框（找不到字型則維持預設、不報錯）。"""
    try:
        import matplotlib
        for font in ["Arial Unicode MS", "PingFang TC", "Heiti TC", "Microsoft JhengHei",
                     "Noto Sans CJK TC", "Noto Sans CJK SC", "WenQuanYi Zen Hei"]:
            if any(font in f.name for f in matplotlib.font_manager.fontManager.ttflist):
                matplotlib.rcParams["font.sans-serif"] = [font]
                matplotlib.rcParams["axes.unicode_minus"] = False
                break
    except Exception:
        pass


def _save_fig(plt, path: str) -> None:
    """存圖並關閉。"""
    plt.savefig(path, dpi=130)
    plt.close()
    print(f"已存圖：{path}")


def run_group_analysis(universe: Optional[list[str]] = None, chart: bool = False,
                       rotation_days: int = 20) -> list[str]:
    """族群輪動分析主流程：印出排行榜，並可選擇輸出圖表。
    族群強度以全股票池計算（不做個股突破/流動性過濾），不需逐檔評分。
    回傳已產生的圖檔路徑清單（未繪圖則為空），方便在 notebook 中直接顯示。"""
    if universe is None:
        universe = build_universe()
    prices = fetch_prices(universe, CFG.lookback_days)
    index_close = fetch_index(CFG.lookback_days)
    industry, names = fetch_company_meta() if CFG.use_industry else ({}, {})
    group_items = build_group_universe(prices, index_close, industry, names)
    print_group_ranking(group_items)
    charts: list[str] = []
    if chart:
        p1 = plot_group_strength(group_items)
        rotation = compute_group_rotation(prices, index_close, industry, rotation_days)
        p2 = plot_group_rotation(rotation)
        charts = [p for p in (p1, p2) if p]
    return charts


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
    parser.add_argument("--no-industry", action="store_true", help="關閉族群（產業）相對強度分析")
    parser.add_argument("--finmind", action="store_true",
                        help="回測時用 FinMind 抓歷史三大法人，讓回測納入籌碼面")
    parser.add_argument("--csv", nargs="?", const="tw_picks.csv", default=None,
                        help="把選股結果匯出成 CSV（可指定檔名，預設 tw_picks.csv）")
    parser.add_argument("--groups", action="store_true",
                        help="輸出族群（產業）強度排行榜")
    parser.add_argument("--chart", action="store_true",
                        help="搭配 --groups：輸出族群強度與輪動趨勢圖（需 matplotlib）")
    parser.add_argument("--no-groups", action="store_true",
                        help="每日選股時不顯示族群強度排行榜")
    args = parser.parse_args()

    if args.threshold is not None:
        CFG.confidence_threshold = args.threshold
    if args.full:
        CFG.use_full_universe = True
    if args.no_sentiment:
        CFG.fetch_sentiment = False
    if args.no_industry:
        CFG.use_industry = False
    if args.finmind:
        CFG.use_finmind = True

    if args.backtest:
        # --full 不適用於回測：全市場池是用「今日」流動性預篩選出的，回放歷史會引入
        # 倖存者偏差與選股未來函數，使勝率失真。改用固定的內建股票池。
        if CFG.use_full_universe:
            print("⚠️ --full 不適用於回測（今日流動性預篩會造成倖存者偏差/選股未來函數），"
                  "改用內建股票池回測。")
        backtest(DEFAULT_UNIVERSE, test_days=args.days)
    elif args.groups:
        run_group_analysis(build_universe(), chart=args.chart)
    else:
        # 每日流程預設先顯示族群排行（族群脈絡），再給選股結果
        picks = run_daily(build_universe(), show_groups=not args.no_groups)
        print_report(picks)
        if args.csv:
            export_csv(picks, args.csv)


if __name__ == "__main__":
    main()
