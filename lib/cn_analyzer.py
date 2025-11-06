# cn_analyzer.py
# -*- coding: utf-8 -*-
"""
A股分析模块（第2步）
- 三种模式：大盘 / 单股 / 多股
- 纯 pandas/numpy 指标实现（EMA/RSI/MACD/ATR/布林/回撤/波动/动量）
- 自适应估值/财务（若有则叠加估值分位、净利增速等）
- 输出：结构化 dict + 精简可读摘要

依赖：
  pip install pandas numpy
（取数走你的 cn_data_fetcher.py，无需额外依赖）
"""

from __future__ import annotations
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# --- 若希望直接跑 CLI，会从你第1步脚本拉数 ---
import cn_data_fetcher as fetch


# ========= 工具层：时序与基础统计 =========
def _as_ts(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    if "date" in out.columns:
        out = out.sort_values("date").set_index("date", drop=True)
    for c in ["open", "high", "low", "close", "volume", "amount"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    # RMA (Wilder)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _macd(close: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd = ema_fast - ema_slow
    macd_signal = _ema(macd, signal)
    hist = macd - macd_signal
    return macd, macd_signal, hist


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def _boll(close: pd.Series, window: int = 20, n_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(window).mean()
    std = close.rolling(window).std(ddof=0)
    upper = mid + n_std * std
    lower = mid - n_std * std
    return mid, upper, lower


def _ret(close: pd.Series, n: int) -> pd.Series:
    return close.pct_change(n)


def _ann_vol(close: pd.Series, win: int = 20) -> float:
    r = close.pct_change().dropna()
    if r.empty:
        return float("nan")
    return float(r.tail(win).std(ddof=0) * np.sqrt(252))


def _max_drawdown(close: pd.Series) -> Tuple[float, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    if close is None or close.dropna().empty:
        return float("nan"), None, None
    c = close.dropna()
    cummax = c.cummax()
    dd = c / cummax - 1.0
    end_idx = dd.idxmin()
    if end_idx is None:
        return 0.0, None, None
    start_idx = c.loc[:end_idx].idxmax()
    return float(dd.min()), start_idx, end_idx


def _slope_annualized(close: pd.Series, window: int = 60) -> float:
    """
    对 log(close) 做简单线性回归斜率，折算为年化（近似趋势强度）。
    """
    c = close.dropna()
    if len(c) < window:
        return float("nan")
    y = np.log(c.tail(window).values)
    x = np.arange(len(y))
    x = (x - x.mean()) / x.std(ddof=0)
    beta = np.polyfit(x, y, 1)[0]
    # 约折算：日为单位 -> *252
    return float(beta * 252)


def _percentile(series: pd.Series, value: float) -> float:
    s = series.dropna()
    if s.empty or value is None or math.isnan(value):
        return float("nan")
    return float((s <= value).mean())


# ========= 指标与打分 =========
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入：含 close/high/low 的 DataFrame（index 为 DatetimeIndex）
    输出：附带 EMA/RSI/MACD/ATR/布林 等列
    """
    ts = _as_ts(df)
    if ts is None or ts.empty:
        return ts
    out = ts.copy()
    out["ema20"] = _ema(out["close"], 20)
    out["ema60"] = _ema(out["close"], 60)
    out["ema120"] = _ema(out["close"], 120)
    out["rsi14"] = _rsi(out["close"], 14)
    macd, macd_sig, macd_hist = _macd(out["close"])
    out["macd"] = macd
    out["macd_signal"] = macd_sig
    out["macd_hist"] = macd_hist
    out["atr14"] = _atr(out["high"], out["low"], out["close"], 14)
    bb_mid, bb_up, bb_lo = _boll(out["close"], 20, 2.0)
    out["bb_mid"] = bb_mid
    out["bb_up"] = bb_up
    out["bb_lo"] = bb_lo
    return out


def trend_snapshot(ts: pd.DataFrame) -> Dict:
    """
    简易趋势判定：
    - 价格与 EMA20/60/120 的相对位置
    - 斜率（年化）
    - MACD 柱
    - 波动与回撤
    """
    if ts is None or ts.empty:
        return {}

    last = ts.iloc[-1]
    price = float(last["close"])
    ema20, ema60, ema120 = float(last["ema20"]), float(last["ema60"]), float(last["ema120"])
    above = sum([price > ema20, price > ema60, price > ema120])

    # 粗略标签
    if above == 3:
        label = "多头"
    elif above == 0:
        label = "空头"
    else:
        label = "震荡"

    slope60 = _slope_annualized(ts["close"], 60)
    vol20 = _ann_vol(ts["close"], 20)
    mdd, mdd_start, mdd_end = _max_drawdown(ts["close"])

    return {
        "label": label,
        "price": price,
        "ema": {"ema20": ema20, "ema60": ema60, "ema120": ema120},
        "slope60_annualized": slope60,
        "ann_vol20": vol20,
        "macd_hist": float(last["macd_hist"]),
        "rsi14": float(last["rsi14"]),
        "max_drawdown": mdd,
        "mdd_start": mdd_start,
        "mdd_end": mdd_end,
    }


def momentum_block(ts: pd.DataFrame, lookbacks=(20, 60, 120, 250)) -> Dict[str, float]:
    out = {}
    for n in lookbacks:
        try:
            r = float(_ret(ts["close"], n).iloc[-1])
        except Exception:
            r = float("nan")
        out[f"ret_{n}d"] = r
    return out


def valuation_block(val_df: Optional[pd.DataFrame], years: int = 5) -> Dict[str, float]:
    """
    对估值做“分位”视角：最近 years 年内 pe/pb/pe_ttm 的分位水平（越低越便宜）
    """
    if val_df is None or val_df.empty:
        return {}

    v = val_df.copy()
    v["date"] = pd.to_datetime(v["date"])
    cutoff = v["date"].max() - pd.Timedelta(days=365 * years)
    v = v[v["date"] >= cutoff]

    last = v.iloc[-1]
    out = {}
    for col in ["pe", "pe_ttm", "pb"]:
        if col in v.columns and pd.notna(last.get(col, np.nan)):
            out[f"{col}_now"] = float(last[col])
            out[f"{col}_pct_{years}y"] = _percentile(v[col], float(last[col]))
    for col in ["dv_ratio"]:
        if col in v.columns and pd.notna(last.get(col, np.nan)):
            out[f"{col}_now"] = float(last[col])
            out[f"{col}_pct_{years}y"] = _percentile(v[col], float(last[col]))
    return out


def quality_block(fin_df: Optional[pd.DataFrame]) -> Dict[str, float]:
    """
    从财务摘要里尽量抓：净利润同比、营收同比、ROE（若能找到）
    字段名在不同源可能略有差异，这里做模糊匹配。
    """
    if fin_df is None or fin_df.empty:
        return {}
    df = fin_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    def _col_like(keys: List[str]) -> Optional[str]:
        for k in df.columns:
            for key in keys:
                if key in str(k):
                    return k
        return None

    col_np = _col_like(["净利润", "净利"])
    col_rev = _col_like(["营业收入", "主营业务收入", "营收"])
    col_roe = _col_like(["ROE", "净资产收益率"])

    out = {}
    # 同比：取最近两期年或中报（差分/上一期）
    def _yoy(col: str) -> Optional[float]:
        nonlocal df
        if col is None or col not in df.columns:
            return None
        tail = df[["date", col]].dropna().tail(5)
        if len(tail) < 2:
            return None
        try:
            v_now = float(tail.iloc[-1][col])
            v_prev = float(tail.iloc[-2][col])
            if v_prev == 0:
                return None
            return (v_now / v_prev) - 1.0
        except Exception:
            return None

    for name, col in [("net_profit_yoy", col_np), ("revenue_yoy", col_rev)]:
        y = _yoy(col)
        if y is not None:
            out[name] = float(y)
    if col_roe and col_roe in df.columns:
        try:
            out["roe_last"] = float(pd.to_numeric(df[col_roe], errors="coerce").dropna().iloc[-1])
        except Exception:
            pass
    return out


def composite_score(trend: Dict, mom: Dict, val: Dict, qual: Dict) -> float:
    """
    简单综合打分（0~100）：趋势/动量/估值/质量
    - 趋势：价格在EMA之上个数、斜率正/负、RSI适中
    - 动量：20/60/120 天收益加权
    - 估值：PE/PB 分位（越低越好）
    - 质量：净利/营收同比、ROE
    """
    score = 0.0

    # 趋势（30分）
    k = 0
    if trend:
        ema_above = sum([trend["price"] > trend["ema"]["ema20"],
                         trend["price"] > trend["ema"]["ema60"],
                         trend["price"] > trend["ema"]["ema120"]])
        k += (ema_above / 3.0) * 20
        slope = trend.get("slope60_annualized", np.nan)
        if not np.isnan(slope):
            k += np.clip((slope / 0.3), -1, 1) * 10  # slope≈0.3/年 视作及格
    score += np.clip(k, 0, 30)

    # 动量（30分）
    m = 0
    w = {"ret_20d": 0.5, "ret_60d": 0.3, "ret_120d": 0.2}
    for key, weight in w.items():
        r = mom.get(key, np.nan)
        if not np.isnan(r):
            m += np.clip(r, -0.3, 0.3) / 0.3 * (30 * weight)
    score += np.clip(m, 0, 30)

    # 估值（25分）——分位越低越好
    v = 0
    for name in ["pe_ttm_pct_5y", "pe_pct_5y", "pb_pct_5y"]:
        p = val.get(name, np.nan)
        if not np.isnan(p):
            v += (1 - p) * (25 / 3.0)
    score += np.clip(v, 0, 25)

    # 质量（15分）
    q = 0
    if "net_profit_yoy" in qual and not np.isnan(qual["net_profit_yoy"]):
        q += np.clip(qual["net_profit_yoy"], -0.5, 0.5) / 0.5 * 6
    if "revenue_yoy" in qual and not np.isnan(qual["revenue_yoy"]):
        q += np.clip(qual["revenue_yoy"], -0.3, 0.3) / 0.3 * 5
    if "roe_last" in qual and not np.isnan(qual["roe_last"]):
        q += np.clip((qual["roe_last"] - 8) / 7, 0, 1) * 4  # ROE 8~15 线性映射
    score += np.clip(q, 0, 15)

    return float(np.clip(score, 0, 100))


# ========= 单股分析 =========
@dataclass
class SingleAnalysis:
    code: str
    trend: Dict
    momentum: Dict
    valuation: Dict
    quality: Dict
    risk: Dict
    score: float


def analyze_single(
    px_df: pd.DataFrame,
    *,
    valuation_df: Optional[pd.DataFrame] = None,
    financial_df: Optional[pd.DataFrame] = None,
    benchmark_df: Optional[pd.DataFrame] = None,
) -> SingleAnalysis:
    ts0 = _as_ts(px_df)
    ts = compute_indicators(ts0)
    trend = trend_snapshot(ts)
    mom = momentum_block(ts)

    val5y = valuation_block(valuation_df, years=5) if valuation_df is not None else {}
    # 给 composite_score 里需要的 key 起固定名
    if val5y:
        if "pe_ttm_pct_5y" not in val5y and "pe_ttm_pct_5y" not in val5y:
            # 兼容：函数里就是以 {col}_pct_{years}y 命名
            pass

    qual = quality_block(financial_df)

    # 风险侧：波动与回撤
    ann_vol20 = _ann_vol(ts["close"], 20)
    mdd, mdd_s, mdd_e = _max_drawdown(ts["close"])
    risk = {"ann_vol20": ann_vol20, "max_drawdown": mdd, "mdd_window": (mdd_s, mdd_e)}

    # 与基准相对强弱（可选）
    if benchmark_df is not None and not benchmark_df.empty:
        b = _as_ts(benchmark_df)
        joined = ts[["close"]].join(b[["close"]].rename(columns={"close": "bench"}), how="inner")
        if not joined.empty:
            rel_60 = float((joined["close"].pct_change(60) - joined["bench"].pct_change(60)).iloc[-1])
            risk["rel_strength_60d_vs_bench"] = rel_60

    # 拼接估值 key 的别名（固定 5 年）
    if val5y:
        # 补 alias，方便总分函数读取
        for base in ["pe", "pe_ttm", "pb", "dv_ratio"]:
            if f"{base}_pct_5y" not in val5y and f"{base}_pct_5y" in val5y:
                pass  # 已存在
        # 无需额外处理（命名一致）

    score = composite_score(trend, mom, val5y, qual)

    return SingleAnalysis(
        code="",
        trend=trend,
        momentum=mom,
        valuation=val5y,
        quality=qual,
        risk=risk,
        score=score,
    )


# ========= 多股分析 =========
@dataclass
class MultiAnalysis:
    ranks: pd.DataFrame        # 动量/波动/分数排名
    corr: Optional[pd.DataFrame]  # 收益相关性（可选）


def analyze_multi(
    long_df: pd.DataFrame,
    *,
    valuation_map: Optional[Dict[str, pd.DataFrame]] = None,
    lookbacks=(20, 60, 120, 250),
) -> MultiAnalysis:
    """
    输入：长表 ['code','date','open'..'close'..]
    输出：每个 code 的动量/波动/估值分位/综合分，并给出排名
    """
    if long_df is None or long_df.empty:
        return MultiAnalysis(pd.DataFrame(), None)

    df = long_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    pivot = df.pivot(index="date", columns="code", values="close").sort_index()

    ranks = []
    for code in pivot.columns:
        s = pivot[code].dropna()
        if len(s) < 40:
            continue
        d = {}
        d["code"] = code
        # 动量
        for n in lookbacks:
            if len(s) > n:
                d[f"ret_{n}d"] = float(_ret(s, n).iloc[-1])
            else:
                d[f"ret_{n}d"] = np.nan
        # 波动/回撤
        d["ann_vol20"] = _ann_vol(s, 20)
        d["mdd"] = _max_drawdown(s)[0]
        # 估值分位（若提供）
        if valuation_map and code in valuation_map:
            v = valuation_block(valuation_map[code], years=5)
            for col in ["pe_ttm_pct_5y", "pe_pct_5y", "pb_pct_5y"]:
                d[col] = v.get(col, np.nan)
        ranks.append(d)

    tab = pd.DataFrame(ranks)
    if tab.empty:
        return MultiAnalysis(pd.DataFrame(), None)

    # 简单打分（与 single 一致思路的子集）
    base_score = pd.Series(0.0, index=tab.index)
    # 动量加权
    for k, w in {"ret_20d": 0.5, "ret_60d": 0.3, "ret_120d": 0.2}.items():
        if k in tab:
            base_score += np.clip(tab[k], -0.3, 0.3) / 0.3 * (30 * w)
    # 估值（分位越低越好）
    val_part = 0
    for k in ["pe_ttm_pct_5y", "pe_pct_5y", "pb_pct_5y"]:
        if k in tab:
            val_part += (1 - tab[k].fillna(0.5)) * (25 / 3.0)
    # 波动/回撤惩罚（越小越好）
    pen = 0
    if "ann_vol20" in tab:
        pen += np.clip((tab["ann_vol20"] - 0.25) / 0.25, 0, 1) * 10  # >25%年化波动扣分
    if "mdd" in tab:
        pen += np.clip((-tab["mdd"] - 0.4) / 0.4, 0, 1) * 10        # 回撤超过40%扣分

    tab["score"] = np.clip(base_score + val_part - pen, 0, 100)
    tab = tab.sort_values("score", ascending=False).reset_index(drop=True)

    # 相关性（最近60日）
    corr = pivot.pct_change().tail(60).corr(min_periods=20)

    return MultiAnalysis(tab, corr)


# ========= 大盘分析 =========
@dataclass
class MarketAnalysis:
    overview: pd.DataFrame     # 各指数：趋势、动量、波动、回撤
    dispersion: float          # 指数间收益离散度（近20日）
    breadth_hint: Optional[Dict]  # 可选：从北向/涨跌家数推断（此处仅北向）
    risk_note: str


def analyze_market(index_long: pd.DataFrame, northbound: Optional[pd.DataFrame] = None) -> MarketAnalysis:
    """
    输入：get_market_indices(..., return_long=True) 的长表
    """
    if index_long is None or index_long.empty:
        return MarketAnalysis(pd.DataFrame(), float("nan"), None, "无数据")

    idx = index_long.copy()
    idx["date"] = pd.to_datetime(idx["date"])
    out_rows = []
    for name, g in idx.groupby("index_name"):
        ts = compute_indicators(_as_ts(g))
        tr = trend_snapshot(ts)
        mom = momentum_block(ts)
        row = {
            "index": name,
            "label": tr.get("label", ""),
            "ret_20d": mom.get("ret_20d", np.nan),
            "ret_60d": mom.get("ret_60d", np.nan),
            "ann_vol20": tr.get("ann_vol20", np.nan),
            "max_dd": tr.get("max_drawdown", np.nan),
            "slope60": tr.get("slope60_annualized", np.nan),
            "rsi14": tr.get("rsi14", np.nan),
        }
        out_rows.append(row)

    overview = pd.DataFrame(out_rows).sort_values("index").reset_index(drop=True)

    # 指数间离散度：近20日收益标准差
    pivot = (idx.pivot(index="date", columns="index_name", values="close").sort_index())
    disp = float(pivot.pct_change(20).iloc[-1].std(ddof=0))

    breadth = None
    if northbound is not None and not northbound.empty:
        nb = northbound.copy()
        nb["date"] = pd.to_datetime(nb["date"])
        recent = nb.tail(20)
        breadth = {
            "north_20d_sum": float(recent["north_net"].sum()) if "north_net" in recent.columns else np.nan,
            "north_last": float(recent["north_net"].iloc[-1]) if "north_net" in recent.columns else np.nan,
        }

    # 风险提示（基于波动/回撤的简单提示）
    high_vol = (overview["ann_vol20"] > 0.28).sum()
    deep_dd = (overview["max_dd"] < -0.35).sum()
    note = "波动偏高" if high_vol >= 2 else "正常"
    if deep_dd >= 2:
        note += " / 历史回撤偏深"

    return MarketAnalysis(overview, disp, breadth, note)


# ========= 友好打印（用于第3步产出会再升级成报告引擎） =========
def single_brief(sa: SingleAnalysis) -> str:
    lines = []
    t = sa.trend
    m = sa.momentum
    v = sa.valuation
    q = sa.quality
    r = sa.risk
    lines.append(f"趋势：{t.get('label','-')} | slope60年化={t.get('slope60_annualized',np.nan):.3f} | RSI14={t.get('rsi14',np.nan):.1f}")
    lines.append(f"动量：20/60/120天={m.get('ret_20d',np.nan):.2%}/{m.get('ret_60d',np.nan):.2%}/{m.get('ret_120d',np.nan):.2%}")
    lines.append(f"风险：年化波动20日={r.get('ann_vol20',np.nan):.2%} | 最大回撤={r.get('max_drawdown',np.nan):.2%}")
    if "rel_strength_60d_vs_bench" in r:
        lines.append(f"相对强弱(60d vs 基准)：{r['rel_strength_60d_vs_bench']:.2%}")
    if v:
        pe = v.get("pe_ttm_now", v.get("pe_now", np.nan))
        pe_q = v.get("pe_ttm_pct_5y", v.get("pe_pct_5y", np.nan))
        pb = v.get("pb_now", np.nan); pb_q = v.get("pb_pct_5y", np.nan)
        lines.append(f"估值：PE≈{pe:.2f}（5年分位={pe_q:.0%}）| PB≈{pb:.2f}（5年分位={pb_q:.0%}）")
    if q:
        lines.append(f"质量：净利YoY={q.get('net_profit_yoy',np.nan):.2%} | 营收YoY={q.get('revenue_yoy',np.nan):.2%} | ROE={q.get('roe_last',np.nan):.2f}%")
    lines.append(f"综合评分：{sa.score:.1f}/100")
    return "\n".join(lines)


# ========= CLI（演示三种模式） =========
def _cli():
    ap = argparse.ArgumentParser(description="第2步：A股分析（三种模式）")
    ap.add_argument("--mode", required=True, choices=["market", "single", "multi"])
    ap.add_argument("--code", help="单股代码，如 600519 或 600519.SH（single 模式必填）")
    ap.add_argument("--codes", help="多股，逗号分隔（multi 模式必填）")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    ap.add_argument("--benchmark", default="000300", help="基准指数代码（默认沪深300，用于单股相对强弱）")
    ap.add_argument("--latest", type=int, default=10)
    args = ap.parse_args()

    if args.mode == "single":
        if not args.code:
            raise SystemExit("single 模式需要 --code")

        # —— 价格（个股）+ 估值 + 财务
        px = fetch.get_stock_daily(args.code, args.start, args.end, period="daily", adjust="qfq")
        val = fetch.get_stock_valuation_hist(args.code)
        fin = fetch.get_stock_financial_abstract(args.code)

        # —— 基准（指数）：你的环境常见缺 amount，这里自动补
        bench = fetch.get_market_indices(
            args.start, args.end, period="daily",
            indices={"沪深300": args.benchmark}, return_long=True
        )
        bench_df = None
        if isinstance(bench, pd.DataFrame) and not bench.empty:
            bd = bench[bench.get("index_code") == args.benchmark].copy()
            if not bd.empty:
                bd["date"] = pd.to_datetime(bd["date"], errors="coerce")
                for col in ["volume", "amount"]:
                    if col not in bd.columns:
                        bd[col] = np.nan
                cols = [c for c in ["date","open","high","low","close","volume","amount"] if c in bd.columns]
                bench_df = bd[cols]

        # —— 基础档案（名称、行业等）
        prof = fetch.get_stock_profile(args.code)

        # —— 打印抬头信息（名称 / 交易所 / 行业 / 最新价 / 52W / YTD）
        hdr_lines = []
        try:
            dfp = px.copy()
            dfp["date"] = pd.to_datetime(dfp["date"], errors="coerce")
            dfp = dfp.sort_values("date").dropna(subset=["date"])
            last_dt = dfp["date"].iloc[-1] if not dfp.empty else None
            last_close = float(dfp["close"].iloc[-1]) if "close" in dfp.columns and not dfp.empty else np.nan
            day_chg = float(dfp["close"].pct_change().iloc[-1]) if "close" in dfp.columns and len(dfp) >= 2 else np.nan

            # 52周窗口（约252交易日）
            win = dfp.tail(252) if len(dfp) >= 2 else dfp
            hi52 = float(win["close"].max()) if "close" in win.columns and not win.empty else np.nan
            lo52 = float(win["close"].min()) if "close" in win.columns and not win.empty else np.nan
            pos52 = float((last_close - lo52) / (hi52 - lo52)) if np.isfinite(hi52) and np.isfinite(lo52) and hi52 > lo52 else np.nan

            # YTD
            if last_dt is not None:
                y0 = pd.Timestamp(year=last_dt.year, month=1, day=1)
                ytd_df = dfp[dfp["date"] >= y0]
                ytd = float(ytd_df["close"].iloc[-1] / ytd_df["close"].iloc[0] - 1.0) if len(ytd_df) >= 2 else np.nan
            else:
                ytd = np.nan

            title = f"[{prof.get('ts_code', '')}] {prof.get('name') or ''}".strip()
            sub = f"交易所: {prof.get('exchange','') or '-'}"
            if prof.get("industry"):
                sub += f" | 行业: {prof['industry']}"
            span = ""
            if not dfp.empty:
                span = f"{dfp['date'].iloc[0]:%Y-%m-%d} → {dfp['date'].iloc[-1]:%Y-%m-%d}（{len(dfp)} bars）"

            hdr_lines.append(title)
            hdr_lines.append(sub)
            if last_dt is not None:
                hdr_lines.append(
                    f"最新({last_dt:%Y-%m-%d}): {last_close:.2f} （日变动 {day_chg:+.2%}）"
                )
            if np.isfinite(hi52) and np.isfinite(lo52):
                hdr_lines.append(
                    f"52W区间: {lo52:.2f} ~ {hi52:.2f} | 当前位置: {pos52:.0%} | YTD: {ytd:.2%}"
                )
            if span:
                hdr_lines.append(f"样本: {span}")
        except Exception:
            pass

        if hdr_lines:
            print("\n".join(hdr_lines))

        # —— 做分析并打印简报
        sa = analyze_single(px, valuation_df=val, financial_df=fin, benchmark_df=bench_df)
        print(single_brief(sa))


    elif args.mode == "multi":
        if not args.codes:
            raise SystemExit("multi 模式需要 --codes")
        cs = [c.strip() for c in args.codes.split(",") if c.strip()]
        long = fetch.get_stocks_batch(cs, args.start, args.end, period="daily", adjust="qfq", return_long=True)

        # 可选：为每个 code 拉估值（可能较慢；按需开启）
        val_map = {}
        for c in cs:
            try:
                val_map[fetch._norm_stock_code(c)] = fetch.get_stock_valuation_hist(c)
            except Exception:
                pass

        ma = analyze_multi(long, valuation_map=val_map)
        print("\n[多股排名（Top 20）]")
        cols = [c for c in ["code", "score", "ret_20d", "ret_60d", "ret_120d", "ann_vol20", "mdd", "pe_ttm_pct_5y", "pb_pct_5y"] if c in ma.ranks.columns]
        if not ma.ranks.empty:
            print(ma.ranks[cols].head(20).to_string(index=False))
        else:
            print("（无可排名数据）")

        if ma.corr is not None:
            print("\n[相关性（近60日，截断前10x10）]")
            small = ma.corr.iloc[:10, :10].round(2)
            print(small.to_string())

    elif args.mode == "market":
        idx = fetch.get_market_indices(args.start, args.end, period="daily", return_long=True)
        nb = fetch.get_northbound_flow(args.start, args.end)
        mk = analyze_market(idx, nb)

        print("\n[大盘概览]")
        if isinstance(mk.overview, pd.DataFrame) and not mk.overview.empty:
            print(mk.overview.to_string(index=False))
        else:
            print("（无概览数据）")

        print(f"\n指数收益离散度(近20日STD)：{mk.dispersion:.4f}")
        if mk.breadth_hint:
            nb_sum = mk.breadth_hint.get('north_20d_sum', float('nan'))
            nb_last = mk.breadth_hint.get('north_last', float('nan'))
            print(f"北向近20日净流入合计：{nb_sum:,.0f}")
            print(f"北向最新净流入：{nb_last:,.0f}")
        print(f"风险提示：{mk.risk_note}")


if __name__ == "__main__":
    _cli()
