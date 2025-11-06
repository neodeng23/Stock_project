#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cn_probe.py — A股数据接口探针（一次性摸清环境 & 列名 & 可用性）
用途：
1) 打印环境信息（Python/akshare/pandas/requests…版本、Tushare 可用性）
2) 探测关键接口的“参数名”和“返回列”：避免猜测（比如财务摘要 symbol/stock）
3) 验证指数/个股主备源是否返回 amount/volume 等列
输出：清晰的可读日志（必要时也会简要显示 head）

用法示例：
  python cn_probe.py --code 600519 --index 000300 --start 2025-01-01
  python cn_probe.py --code 600519 --index 000001 --start 2024-10-01 --end 2025-11-06
"""

import sys, os, inspect, traceback, argparse, datetime as dt
from typing import Optional
import pandas as pd

def _p(msg=""):
    print(msg, flush=True)

def _hdr(title):
    _p("\n" + "="*8 + f" {title} " + "="*8)

def _show_df(df: Optional[pd.DataFrame], name: str, rows: int = 3):
    if df is None:
        _p(f"[{name}] -> None")
        return
    if not isinstance(df, pd.DataFrame):
        _p(f"[{name}] -> {type(df)}")
        return
    _p(f"[{name}] shape={df.shape}")
    if df.empty:
        _p(f"[{name}] empty DataFrame, columns={list(df.columns)}")
        return
    _p(f"[{name}] columns={list(df.columns)}")
    head = df.head(rows)
    # 尽量打印关键列是否存在
    cols_interest = [c for c in ["date","open","high","low","close","volume","amount","截止日期","报告日期","报告期","公告日期","日期"] if c in df.columns]
    if cols_interest:
        _p(f"[{name}] key columns sample:\n{head[cols_interest]}")
    else:
        _p(f"[{name}] head:\n{head}")

def _sig_of(obj):
    try:
        return str(inspect.signature(obj))
    except Exception:
        return "<signature unavailable>"

def _try(callable_or_lambda, label: str):
    """执行一次调用并捕获异常，返回 DataFrame 或 None；异常打印栈摘要。"""
    try:
        res = callable_or_lambda()
        return res
    except Exception as e:
        _p(f"[{label}] EXC: {e.__class__.__name__}: {e}")
        tb = "".join(traceback.format_exception_only(type(e), e)).strip()
        _p(f"[{label}] brief: {tb}")
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True, help="股票代码：6位或带.SH/.SZ，如 600519 或 600519.SH")
    ap.add_argument("--index", default="000300", help="指数代码：000300(沪深300)/000001(上证综指)/399001 等")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default=dt.date.today().strftime("%Y-%m-%d"))
    args = ap.parse_args()

    # 统一代码形态
    raw = args.code.strip().upper()
    code6 = "".join([c for c in raw if c.isdigit()])[:6]
    if not code6:
        _p("无效股票代码"); sys.exit(2)
    def prefixed(c6: str) -> str:
        if c6.startswith(("6","9")): return "sh"+c6
        if c6.startswith(("0","3")): return "sz"+c6
        if c6.startswith(("4","8")): return "bj"+c6
        return c6

    _hdr("环境信息")
    _p(f"Python: {sys.version.split()[0]}")
    try:
        import akshare as ak, pandas as pd, numpy as np, requests, urllib3
        _p(f"akshare: {getattr(ak, '__version__', 'unknown')}")
        _p(f"pandas: {pd.__version__}")
        _p(f"numpy: {np.__version__}")
        _p(f"requests: {requests.__version__}")
        _p(f"urllib3: {urllib3.__version__}")
    except Exception as e:
        _p(f"导入依赖失败: {e}")
        sys.exit(2)

    tushare_ok = False
    try:
        import tushare as ts  # 可选
        tok = os.getenv("TUSHARE_TOKEN")
        _p(f"Tushare installed: yes, TOKEN set: {'yes' if tok else 'no'}")
        tushare_ok = bool(tok)
    except Exception:
        _p("Tushare installed: no")

    # 1) 财务摘要接口形态
    _hdr("财务摘要接口探测 ak.stock_financial_abstract")
    import akshare as ak
    _p(f"signature: stock_financial_abstract{_sig_of(ak.stock_financial_abstract)}")

    sym6 = code6
    sym_pref = prefixed(code6)

    # 依次尝试：symbol=六位；symbol=前缀；位置参数六位；位置参数前缀；（旧参数名 stock=… 也测一下）
    df1 = _try(lambda: ak.stock_financial_abstract(symbol=sym6), "financial symbol=sym6")
    _show_df(df1, "fin symbol=sym6")
    if df1 is None or df1.empty:
        df2 = _try(lambda: ak.stock_financial_abstract(symbol=sym_pref), "financial symbol=pref")
        _show_df(df2, "fin symbol=pref")
    df3 = _try(lambda: ak.stock_financial_abstract(sym6), "financial positional sym6")
    _show_df(df3, "fin positional sym6")
    df4 = _try(lambda: ak.stock_financial_abstract(sym_pref), "financial positional pref")
    _show_df(df4, "fin positional pref")
    # 旧参数名试探（如果报 TypeError，说明不支持）
    df5 = _try(lambda: ak.stock_financial_abstract(stock=sym6), "financial stock=sym6 [legacy?]")
    if df5 is not None:
        _show_df(df5, "fin stock=sym6")

    # 2) 指数：两条路线，观察是否含 amount 列
    _hdr("指数接口探测（是否有 amount 列）")
    s = args.start.replace("-", "")
    e = args.end.replace("-", "")
    df_idx1 = _try(lambda: ak.index_zh_a_hist(symbol=args.index, period="daily", start_date=s, end_date=e), "index_zh_a_hist")
    _show_df(df_idx1, "index_zh_a_hist")
    idx_pref = ("sh" if args.index.startswith(("0","5")) else "sz") + args.index
    df_idx2 = _try(lambda: ak.stock_zh_index_daily(symbol=idx_pref), "stock_zh_index_daily")
    # 为便于比较，取区间
    if isinstance(df_idx2, pd.DataFrame) and not df_idx2.empty and "日期" in df_idx2.columns:
        df_idx2 = df_idx2[(df_idx2["日期"] >= args.start) & (df_idx2["日期"] <= args.end)]
    _show_df(df_idx2, "stock_zh_index_daily")

    # 3) 个股：主路（东财）与备用（新浪），看是否可用 & 列名
    _hdr("个股K线探测（主/备源）")
    df_eq1 = _try(lambda: ak.stock_zh_a_hist(symbol=sym6, period="daily", start_date=s, end_date=e, adjust="qfq"), "stock_zh_a_hist (eastmoney)")
    _show_df(df_eq1, "stock_zh_a_hist")
    df_eq2 = _try(lambda: ak.stock_zh_a_daily(symbol=sym_pref, adjust="qfq"), "stock_zh_a_daily (sina)")
    # 截区间
    if isinstance(df_eq2, pd.DataFrame) and not df_eq2.empty and "date" in df_eq2.rename(columns={"日期":"date"}).columns:
        tmp = df_eq2.rename(columns={"日期":"date"})
        tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
        m1, m2 = pd.to_datetime(args.start), pd.to_datetime(args.end)
        df_eq2 = tmp[(tmp["date"] >= m1) & (tmp["date"] <= m2)]
    _show_df(df_eq2, "stock_zh_a_daily")

    # 4) 若 Tushare 可用，简单探测日线 & 指数 & 日常估值
    if tushare_ok:
        _hdr("Tushare 探测")
        import tushare as ts
        ts.set_token(os.getenv("TUSHARE_TOKEN"))
        pro = ts.pro_api()
        ts_code = (code6 + ".SH") if code6.startswith(("6","9")) else (code6 + ".SZ")
        d1 = _try(lambda: pro.daily(ts_code=ts_code, start_date=s, end_date=e), "ts.daily")
        _show_df(d1, "ts.daily")
        i_code = args.index + (".SH" if args.index.startswith(("0","5")) else ".SZ")
        d2 = _try(lambda: pro.index_daily(ts_code=i_code, start_date=s, end_date=e), "ts.index_daily")
        _show_df(d2, "ts.index_daily")
        d3 = _try(lambda: pro.daily_basic(ts_code=ts_code, fields="trade_date,pe,pe_ttm,pb,total_mv,circ_mv"), "ts.daily_basic")
        _show_df(d3, "ts.daily_basic")
    else:
        _p("\n[Tushare] 未安装或未设置 TOKEN，跳过探测。")

    _p("\n探测结束：请把整段输出贴给我，我会基于你本机接口形态一次性给出稳妥改动。")

if __name__ == "__main__":
    main()
