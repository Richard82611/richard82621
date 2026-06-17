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
