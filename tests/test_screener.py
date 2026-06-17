# -*- coding: utf-8 -*-
"""
tw_momentum_screener 的離線測試（不需網路）。
用合成資料 + stub 驗證核心邏輯：技術指標、評分、族群相對強度、
FinMind 解析、風控與 CSV 匯出。CI 與本機皆可執行：

    python tests/test_screener.py
"""

import os
import sys
import types

# --- 讓測試免裝/免連網：先 stub yfinance，再把專案根目錄加入路徑 ---
_fake_yf = types.ModuleType("yfinance")
_fake_yf.download = lambda *a, **k: None
sys.modules.setdefault("yfinance", _fake_yf)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import tw_momentum_screener as m  # noqa: E402


def test_weights_sum_to_one():
    c = m.CFG
    total = (c.w_rs + c.w_group + c.w_breakout + c.w_trend
             + c.w_momentum + c.w_chips + c.w_sentiment)
    assert abs(total - 1.0) < 1e-9, f"權重總和應為 1.0，實際 {total}"


def test_rsi_edge_cases():
    up = pd.Series(np.arange(1, 60, dtype=float))      # 全漲：loss=0
    assert abs(m.rsi(up).iloc[-1] - 100.0) < 1e-6, "全漲 RSI 應為 100"
    flat = pd.Series([10.0] * 60)                      # 無波動：gain=loss=0
    assert abs(m.rsi(flat).iloc[-1] - 50.0) < 1e-6, "無波動 RSI 應為中性 50"


def test_market_of():
    assert m.market_of("6488.TWO") == "上櫃"
    assert m.market_of("2330.TW") == "上市"


def test_universe_has_both_markets_and_no_dupes():
    u = m.DEFAULT_UNIVERSE
    assert any(x.endswith(".TW") for x in u), "需含上市"
    assert any(x.endswith(".TWO") for x in u), "需含上櫃"
    assert len(u) == len(set(u)), "股票池不應有重複代碼"


def _make_breakout_df(n=260, seed=0):
    idx = pd.bdate_range(end="2026-06-15", periods=n)
    close = np.linspace(20, 50, n) + np.random.RandomState(seed).normal(0, 0.3, n)
    close[-1] = close[-2] * 1.06  # 今日突破 +6%
    df = pd.DataFrame({
        "Open": close * 0.999, "High": close * 1.02, "Low": close * 0.98,
        "Close": close, "Volume": np.full(n, 3e6),
    }, index=idx)
    df.iloc[-1, df.columns.get_loc("Volume")] = 9e6  # 爆量
    return df


def test_compute_features_and_atr():
    df = _make_breakout_df()
    ic = pd.Series(np.linspace(15000, 16000, len(df)), index=df.index)
    s = m.compute_features("6488.TWO", df, ic, {}, {})
    assert s is not None, "突破股應通過硬性過濾"
    assert s.atr > 0 and s.stop_loss < s.close < s.take_profit, "ATR 停損停利需夾住收盤價"
    assert s.market == "上櫃"


def test_group_relative_strength():
    def mk(sym, rs, ind):
        s = m.StockScore(sym, 0, 50.0)
        s.rs_score = rs
        s.industry = ind
        s.breakout_score = s.trend_score = s.momentum_score = 50.0
        s.chips_score = s.sentiment_score = 50.0
        return s

    a = mk("2330.TW", 0.90, "半導體")   # 強勢族群帶頭
    b = mk("2454.TW", 0.80, "半導體")
    c = mk("2002.TW", 0.10, "鋼鐵")
    d = mk("2013.TW", 0.05, "鋼鐵")
    m.finalize_scores([a, b, c, d])

    assert a.group_score > c.group_score, "強勢族群分數應高於弱勢族群"
    assert a.is_group_leader and not b.is_group_leader, "2330 應為族群帶頭股"


def test_finmind_parsing(monkeypatch_get=None):
    data = {"status": 200, "data": [
        {"date": "2024-01-02", "stock_id": "2330", "buy": 5_000_000, "sell": 1_000_000, "name": "Foreign"},
        {"date": "2024-01-02", "stock_id": "2330", "buy": 2_000_000, "sell": 500_000, "name": "Trust"},
        {"date": "2024-01-03", "stock_id": "2330", "buy": 1_000_000, "sell": 3_000_000, "name": "Foreign"},
    ]}

    class _Resp:
        def json(self):
            return data

    saved = m.requests
    m.requests = types.SimpleNamespace(get=lambda *a, **k: _Resp())
    try:
        res = m.fetch_finmind_chips(["2330.TW"], "2024-01-01", "2024-01-31")
    finally:
        m.requests = saved

    assert abs(res["2330"]["2024-01-02"] - 5500.0) < 1e-6, "01-02 淨買超應為 5500 張"
    assert abs(res["2330"]["2024-01-03"] + 2000.0) < 1e-6, "01-03 淨賣超應為 -2000 張"


def test_headline_sentiment():
    assert m._score_headlines(["台積電大漲創新高 法人買超"]) > 0
    assert m._score_headlines(["某股大跌跌停 認列損失"]) < 0
    assert m._score_headlines(["公司召開股東會"]) == 0


def test_rank_groups():
    def mk(sym, rs, ind):
        s = m.StockScore(sym, 0, 50.0)
        s.rs_score = rs
        s.industry = ind
        s.breakout_score = s.trend_score = s.momentum_score = 50.0
        s.chips_score = s.sentiment_score = 50.0
        return s

    scores = [mk("2330.TW", 0.95, "半導體"), mk("2454.TW", 0.85, "半導體"),
              mk("2603.TW", 0.50, "航運"),
              mk("2002.TW", 0.10, "鋼鐵"), mk("2013.TW", 0.05, "鋼鐵")]
    m.finalize_scores(scores)
    rows = m.rank_groups(scores)
    assert rows[0]["industry"] == "半導體", "最強族群應排第一"
    assert rows[-1]["industry"] == "鋼鐵", "最弱族群應排最後"
    assert rows[0]["leader"] == "2330.TW", "半導體帶頭股應為 2330"


def test_group_rotation():
    n = 160
    idx = pd.bdate_range(end="2026-06-15", periods=n)

    def price(slope, seed):
        c = np.linspace(20, 20 + slope, n) + np.random.RandomState(seed).normal(0, 0.2, n)
        return pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99,
                             "Close": c, "Volume": np.full(n, 3e6)}, index=idx)

    prices = {"2330.TW": price(40, 1), "2454.TW": price(35, 2),
              "2603.TW": price(8, 3), "2002.TW": price(-2, 4)}
    industry = {"2330": "半導體", "2454": "半導體", "2603": "航運", "2002": "鋼鐵"}
    ic = pd.Series(np.linspace(15000, 16000, n), index=idx)
    rot = m.compute_group_rotation(prices, ic, industry, lookback_days=15)
    assert not rot.empty and rot.shape[0] > 0, "輪動表應有資料"
    assert "半導體" in rot.columns, "輪動表應含半導體欄"
    # 強勢族群最新強度應高於弱勢族群
    assert rot["半導體"].iloc[-1] > rot["鋼鐵"].iloc[-1], "半導體最新強度應高於鋼鐵"


def test_csv_export(tmp_path_str="/tmp/_tw_test_out.csv"):
    df = _make_breakout_df()
    ic = pd.Series(np.linspace(15000, 16000, len(df)), index=df.index)
    s = m.compute_features("6488.TWO", df, ic, {}, {})
    m.finalize_scores([s])
    path = m.export_csv([s], tmp_path_str)
    assert path and os.path.exists(path), "CSV 應成功產出"
    header = open(path, encoding="utf-8-sig").readline()
    assert "產業" in header and "停損" in header, "CSV 應含產業與停損欄"
    os.remove(path)


def test_percentile_rank_ties():
    """相同 RS 值應得到相同百分位（不因輸入順序而異）。"""
    pct = m._percentile_rank([0.0, 0.0, 0.0, 0.9])
    assert pct[0] == pct[1] == pct[2], "相同數值應同百分位"
    assert pct[3] == 100.0, "最大值應為 100 百分位"


def test_group_universe_ignores_hard_filters():
    """族群強度應涵蓋全股票池（含低流動性個股），不受個股硬性過濾影響。"""
    n = 200
    idx = pd.bdate_range(end="2026-06-15", periods=n)

    def price(level, slope):
        c = np.linspace(level, level + slope, n)
        return pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99,
                             "Close": c, "Volume": np.full(n, 5e5)}, index=idx)  # 低量

    prices = {"1111.TW": price(50, 30), "1112.TW": price(48, 28),
              "2221.TW": price(30, -5), "2222.TW": price(30, -6)}
    industry = {"1111": "半導體", "1112": "半導體", "2221": "鋼鐵", "2222": "鋼鐵"}
    ic = pd.Series(np.linspace(15000, 15500, n), index=idx)
    gi = m.build_group_universe(prices, ic, industry)
    assert len(gi) == 4, "全市場族群清單應含所有有足夠歷史者，不受流動性過濾"
    rows = m.rank_groups(gi)
    assert rows[0]["industry"] == "半導體", "強勢族群應排第一"


def test_volume_baseline_excludes_today():
    """量能基準應排除當日：今日量 = 前 20 日均量的 1.5 倍應觸發爆量加分。"""
    n = 260
    idx = pd.bdate_range(end="2026-06-15", periods=n)
    close = np.linspace(20, 50, n)
    close[-1] = close[-2] * 1.06
    vol = np.full(n, 2_000_000.0)
    vol[-1] = 3_000_000.0  # 恰為前 20 日均量(2.0M)的 1.5 倍
    df = pd.DataFrame({"Open": close * 0.999, "High": close * 1.02, "Low": close * 0.98,
                       "Close": close, "Volume": vol}, index=idx)
    ic = pd.Series(np.linspace(15000, 16000, n), index=idx)
    s = m.compute_features("2330.TW", df, ic, {}, {})
    assert s is not None and any("爆量" in r for r in s.reasons), "1.5x 前20日均量應觸發爆量"


def test_normalize_industry():
    assert m._normalize_industry("24") == "半導體業", "代碼 24 應為半導體業"
    assert m._normalize_industry("10") == "鋼鐵工業"
    assert m._normalize_industry("半導體業") == "半導體業", "名稱應原樣保留"
    assert m._normalize_industry("99") == "99", "未知代碼維持原值"


def test_finalize_uses_full_universe_rs():
    """提供全市場 RS map 時，個股 RS 不因存活檔數少而被重排。"""
    a = m.StockScore("AAA.TW", 0, 50.0)
    a.rs_score = 0.0
    a.industry = "半導體業"
    a.breakout_score = a.trend_score = a.momentum_score = 50.0
    a.chips_score = a.sentiment_score = 50.0
    m.finalize_scores([a], rs_pct_map={"AAA.TW": 95.0},
                      group_strength={"半導體業": 88.0},
                      group_leaders={"半導體業": "AAA.TW"})
    assert a.rs_score == 95.0, "應採全市場 RS 百分位"
    assert a.group_score == 88.0 and a.is_group_leader


def test_full_universe_excludes_etfs():
    """全市場掃描應排除 00 開頭的 ETF（0050/0056）。"""
    class _Resp:
        def __init__(self, d):
            self._d = d

        def json(self):
            return self._d

    rows = [{"Code": "2330", "Name": "台積電", "ClosingPrice": "900", "TradeVolume": "30000000"},
            {"Code": "0050", "Name": "元大台灣50", "ClosingPrice": "180", "TradeVolume": "20000000"},
            {"Code": "0056", "Name": "高股息", "ClosingPrice": "38", "TradeVolume": "50000000"}]
    saved = m.requests
    m.requests = types.SimpleNamespace(get=lambda *a, **k: _Resp(rows))
    try:
        syms = [r["symbol"] for r in m.fetch_twse_all_day()]
    finally:
        m.requests = saved
    assert "2330.TW" in syms, "普通股應保留"
    assert "0050.TW" not in syms and "0056.TW" not in syms, "ETF 應被排除"


def test_fetch_prices_drops_stale():
    """fetch_prices 應剔除最後一根 K 棒明顯落後整批的停滯個股。"""
    n = 80
    fresh = pd.bdate_range(end="2026-06-15", periods=n)
    stale = pd.bdate_range(end="2026-05-01", periods=n)

    def frame(idx):
        c = np.linspace(20, 30, n)
        return pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99,
                             "Close": c, "Volume": np.full(n, 3e6)}, index=idx)

    def fake_dl(symbols, **k):
        return pd.concat({"FRESH.TW": frame(fresh), "STALE.TW": frame(stale)}, axis=1)

    saved = sys.modules["yfinance"].download
    sys.modules["yfinance"].download = fake_dl
    try:
        out = m.fetch_prices(["FRESH.TW", "STALE.TW"], 260)
    finally:
        sys.modules["yfinance"].download = saved
    assert "FRESH.TW" in out and "STALE.TW" not in out, "停滯資料應被剔除"


def main():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ✅ {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ❌ {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} 測試通過。")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
