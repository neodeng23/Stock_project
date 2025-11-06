# cn_data_fetcher.py
# -*- coding: utf-8 -*-
"""
A股数据获取（多提供商 + 自动兜底），尽量提高稳定性与数据维度
依赖（按需）：
  pip install akshare pandas numpy
  # 可选增强：
  pip install tushare       # 若使用 Tushare，需要环境变量 TUSHARE_TOKEN
  pip install baostock      # 作为最终兜底

功能概览：
  价格：get_stock_daily（多源兜底；日/周/月）、get_stocks_batch、get_market_indices（指数多源兜底）
  估值：get_stock_valuation_hist（Tushare daily_basic / AkShare / 百度兜底）
  财务：get_stock_financial_abstract（AkShare·新浪）
  分红：get_stock_bonus_dividend（AkShare）
  分钟：get_stock_minute_bars（AkShare）
  资金：get_northbound_flow（AkShare）
  日历：get_trade_calendar（AkShare·新浪）
  实用：as_timeseries / pivot_close / merge_price_with_valuation / assert_ready_for_ta

统一价格字段：date, open, high, low, close, volume, amount
"""

from __future__ import annotations
import os
import re
import time
from typing import Iterable, Dict, List, Optional, Callable

import numpy as np
import pandas as pd

# 可选库：存在才启用
try:
    import tushare as ts  # type: ignore
except Exception:
    ts = None  # noqa: N816
try:
    import baostock as bs  # type: ignore
except Exception:
    bs = None

import akshare as ak  # 主力与若干备用接口


# ========== 通用工具 ==========
def _to_ak_date(s: str) -> str:
    s = str(s).strip()
    if re.fullmatch(r"\d{8}", s): return s
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s): return s.replace("-", "")
    raise ValueError(f"非法日期格式：{s}，期望 'YYYYMMDD' 或 'YYYY-MM-DD'")


def _norm_stock_code(code: str) -> str:
    c = code.strip().upper().replace("SH", "").replace("SZ", "").replace("BJ", "").replace(".", "")
    m = re.search(r"(\d{6})", c)
    if not m: raise ValueError(f"无法识别股票代码：{code}")
    return m.group(1)


def _mk_prefixed_code(code6: str) -> str:
    c = str(code6)
    if c.startswith(("6","9")): return "sh"+c
    if c.startswith(("0","3")): return "sz"+c
    if c.startswith(("4","8")): return "bj"+c
    return c


def _to_ts_code(code6: str) -> str:
    """Tushare ts_code：600519 -> 600519.SH / 000001 -> 000001.SZ / 430xxx -> .BJ"""
    c = str(code6)
    if c.startswith(("6","9")): return f"{c}.SH"
    if c.startswith(("0","3")): return f"{c}.SZ"
    if c.startswith(("4","8")): return f"{c}.BJ"
    return c


def _rename_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "日期":"date","开盘":"open","最高":"high","最低":"low","收盘":"close",
        "成交量":"volume","成交额":"amount",
        # 英文列保持不变
    }
    df = df.rename(columns=mapping)
    keep = [c for c in ["date","open","high","low","close","volume","amount"] if c in df.columns]
    return df[keep].copy()


def _with_retry(fn: Callable, retries: int=5, base_sleep: float=0.8):
    last = None
    for i in range(retries):
        try:
            return fn()
        except Exception as e:
            last = e
            time.sleep(base_sleep * (i+1))
    raise last


def get_stock_profile(code: str) -> dict:
    """
    返回基础档案：name、industry、exchange、list_date、code6、ts_code
    以 ak.stock_individual_info_em 为主；兜底用 ak.stock_zh_a_spot_em 做名称映射。
    """
    import pandas as pd
    sym6 = _norm_stock_code(code)

    # 推交易所、ts_code
    if sym6.startswith(("6", "9")):
        exchange = "SH"
        ts_code = f"{sym6}.SH"
    elif sym6.startswith(("0", "3")):
        exchange = "SZ"
        ts_code = f"{sym6}.SZ"
    elif sym6.startswith(("4", "8")):
        exchange = "BJ"
        ts_code = f"{sym6}.BJ"
    else:
        exchange = ""
        ts_code = sym6

    name = None
    industry = None
    list_date = None

    # 主路：个股信息表（item/value 形式）
    try:
        df = ak.stock_individual_info_em(symbol=sym6)
        if isinstance(df, pd.DataFrame) and not df.empty:
            # 统一两列名
            cols = {c.lower(): c for c in df.columns}
            item_col = cols.get("item") or cols.get("指标") or list(df.columns)[0]
            val_col  = cols.get("value") or cols.get("值") or list(df.columns)[1]
            m = df.set_index(item_col)[val_col].astype(str)

            # 常见键名的模糊匹配
            def pick(keys):
                for k in m.index:
                    for kk in keys:
                        if kk in str(k):
                            return str(m.loc[k])
                return None

            name = pick(["公司名称", "证券简称", "股票简称"]) or name
            industry = pick(["所属行业", "行业"]) or industry
            ld = pick(["上市日期", "挂牌日期"])
            if ld:
                try:
                    list_date = pd.to_datetime(ld, errors="coerce")
                except Exception:
                    list_date = None
    except Exception:
        pass

    # 兜底：现价快照里映射名称
    if not name:
        try:
            spot = ak.stock_zh_a_spot_em()
            if isinstance(spot, pd.DataFrame) and not spot.empty:
                # 代码列可能是“代码”或“symbol”，名称列可能是“名称”或“name”
                code_col = "代码" if "代码" in spot.columns else ("symbol" if "symbol" in spot.columns else None)
                name_col = "名称" if "名称" in spot.columns else ("name" if "name" in spot.columns else None)
                if code_col and name_col:
                    row = spot[spot[code_col].astype(str).str[-6:] == sym6]
                    if not row.empty:
                        name = str(row.iloc[0][name_col])
        except Exception:
            pass

    return {
        "code6": sym6,
        "ts_code": ts_code,
        "exchange": exchange,
        "name": name or "",
        "industry": industry or "",
        "list_date": list_date,
    }


# ========== 单股 K 线：各提供商实现 ==========
def _get_stock_daily_tushare(code6: str, start: str, end: str, *, period: str, adjust: str) -> pd.DataFrame:
    """Tushare：日/周/月；需要 TUSHARE_TOKEN"""
    if ts is None: 
        raise RuntimeError("tushare 未安装")
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("未设置 TUSHARE_TOKEN")
    ts.set_token(token)
    pro = ts.pro_api()

    ts_code = _to_ts_code(code6)
    s, e = _to_ak_date(start), _to_ak_date(end)

    # Tushare 的复权通过复权因子实现，这里简单调用 pro_bar（需一定权限）
    # 如因权限限制失败，则退到 daily + 你自行前复权（此处直接返回不复权）
    try:
        df = ts.pro_bar(ts_code=ts_code, start_date=s, end_date=e, freq={"daily":"D","weekly":"W","monthly":"M"}[period], adj={"":"", "qfq":"qfq","hfq":"hfq"}[adjust] or None)
        if isinstance(df, pd.DataFrame) and not df.empty:
            df.rename(columns={"trade_date":"date","open":"open","high":"high","low":"low","close":"close","vol":"volume","amount":"amount"}, inplace=True)
            df["date"] = pd.to_datetime(df["date"])
            df = df[["date","open","high","low","close","volume","amount"]].sort_values("date").reset_index(drop=True)
            return df
    except Exception:
        pass

    # 退化到 daily（日线不复权；周/月在本地重采样）
    db = pro.daily(ts_code=ts_code, start_date=s, end_date=e)
    if not isinstance(db, pd.DataFrame) or db.empty:
        return pd.DataFrame()
    db.rename(columns={"trade_date":"date","open":"open","high":"high","low":"low","close":"close","vol":"volume","amount":"amount"}, inplace=True)
    db["date"] = pd.to_datetime(db["date"])
    db = db[["date","open","high","low","close","volume","amount"]].sort_values("date").reset_index(drop=True)

    if period == "daily":
        return db

    tsdf = db.set_index("date").sort_index()
    rule = "W-FRI" if period=="weekly" else "M"
    parts = [
        tsdf["open"].resample(rule).first(),
        tsdf["high"].resample(rule).max(),
        tsdf["low"].resample(rule).min(),
        tsdf["close"].resample(rule).last(),
    ]
    names = ["open","high","low","close"]
    if "volume" in tsdf.columns:
        parts.append(tsdf["volume"].resample(rule).sum(min_count=1)); names.append("volume")
    if "amount" in tsdf.columns:
        parts.append(tsdf["amount"].resample(rule).sum(min_count=1)); names.append("amount")
    out = (pd.concat(parts, axis=1).set_axis(names, axis=1)
           .dropna(subset=["open","high","low","close"]).reset_index())
    return out


def _fetch_stock_daily_fallback_ak_sina(code6: str, start: str, end: str, *, period: str, adjust: str) -> pd.DataFrame:
    """AkShare·新浪：作为东财失败时的备用源"""
    sym_pref = _mk_prefixed_code(code6)
    try:
        df = ak.stock_zh_a_daily(symbol=sym_pref, adjust=adjust)
    except Exception:
        df = None
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=["date","open","high","low","close","volume","amount"])

    df = _rename_ohlc(df)
    if "date" in df.columns: df["date"] = pd.to_datetime(df["date"])
    for c in ["open","high","low","close","volume","amount"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open","high","low","close"]).sort_values("date").reset_index(drop=True)

    s, e = pd.to_datetime(_to_ak_date(start)), pd.to_datetime(_to_ak_date(end))
    df = df[(df["date"] >= s) & (df["date"] <= e)]

    if period == "daily":
        return df.reset_index(drop=True)

    tsdf = df.set_index("date").sort_index()
    rule = "W-FRI" if period=="weekly" else "M"
    parts = [
        tsdf["open"].resample(rule).first(),
        tsdf["high"].resample(rule).max(),
        tsdf["low"].resample(rule).min(),
        tsdf["close"].resample(rule).last(),
    ]
    names = ["open","high","low","close"]
    if "volume" in tsdf.columns:
        parts.append(tsdf["volume"].resample(rule).sum(min_count=1)); names.append("volume")
    if "amount" in tsdf.columns:
        parts.append(tsdf["amount"].resample(rule).sum(min_count=1)); names.append("amount")
    out = (pd.concat(parts, axis=1).set_axis(names, axis=1)
           .dropna(subset=["open","high","low","close"]).reset_index())
    return out


def _get_stock_daily_akshare(code6: str, start: str, end: str, *, period: str, adjust: str) -> pd.DataFrame:
    """AkShare·东财主路 + 新浪兜底"""
    s, e = _to_ak_date(start), _to_ak_date(end)
    def _primary():
        return ak.stock_zh_a_hist(symbol=code6, period=period, start_date=s, end_date=e, adjust=adjust)
    try:
        df = _with_retry(_primary, retries=5, base_sleep=0.8)
    except Exception:
        df = None
    if isinstance(df, pd.DataFrame) and not df.empty:
        df = _rename_ohlc(df)
        if "date" in df.columns: df["date"] = pd.to_datetime(df["date"])
        for c in ["open","high","low","close","volume","amount"]:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.dropna(subset=["open","high","low","close"]).sort_values("date").reset_index(drop=True)

    # 备用：新浪
    return _fetch_stock_daily_fallback_ak_sina(code6, start, end, period=period, adjust=adjust)


def _get_stock_daily_baostock(code6: str, start: str, end: str, *, period: str, adjust: str) -> pd.DataFrame:
    """Baostock：免费兜底（支持日K；周/月本地重采样；复权采用其自带参数）"""
    if bs is None:
        raise RuntimeError("baostock 未安装")
    lg = bs.login()
    try:
        # baostock 复权：'3'前复权、'1'后复权、'2'不复权
        qfq_map = {"qfq":"3","hfq":"1","": "2"}
        rs = bs.query_history_k_data_plus(
            f"sh.{code6}" if code6.startswith(("6","9")) else f"sz.{code6}",
            "date,open,high,low,close,volume,amount",
            start_date=start, end_date=end,
            frequency="d", adjustflag=qfq_map.get(adjust, "2")
        )
        data = []
        while (rs is not None) and rs.error_code == '0' and rs.next():
            data.append(rs.get_row_data())
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=rs.fields)
        df.rename(columns={"date":"date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        for c in ["open","high","low","close","volume","amount"]:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["open","high","low","close"]).sort_values("date").reset_index(drop=True)

        if period == "daily":
            return df

        tsdf = df.set_index("date").sort_index()
        rule = "W-FRI" if period=="weekly" else "M"
        parts = [
            tsdf["open"].resample(rule).first(),
            tsdf["high"].resample(rule).max(),
            tsdf["low"].resample(rule).min(),
            tsdf["close"].resample(rule).last(),
            tsdf["volume"].resample(rule).sum(min_count=1),
            tsdf["amount"].resample(rule).sum(min_count=1),
        ]
        out = (pd.concat(parts, axis=1).set_axis(["open","high","low","close","volume","amount"], axis=1)
               .dropna(subset=["open","high","low","close"]).reset_index())
        return out
    finally:
        bs.logout()


# ========== 统一入口：单股 K 线（多源兜底） ==========
def get_stock_daily(code: str, start: str, end: str, *, period: str="daily", adjust: str="qfq",
                    providers_order: Optional[List[str]]=None) -> pd.DataFrame:
    """
    providers_order: eg. ["tushare","akshare","baostock"]
    未指定时自动：若设置了 TUSHARE_TOKEN -> tushare优先，否则 akshare -> baostock
    """
    sym6 = _norm_stock_code(code)
    if providers_order is None:
        providers_order = ["tushare","akshare","baostock"] if os.getenv("TUSHARE_TOKEN") else ["akshare","baostock","tushare"]

    last_err = None
    for prov in providers_order:
        try:
            if prov == "tushare":
                return _get_stock_daily_tushare(sym6, start, end, period=period, adjust=adjust)
            if prov == "akshare":
                return _get_stock_daily_akshare(sym6, start, end, period=period, adjust=adjust)
            if prov == "baostock":
                return _get_stock_daily_baostock(sym6, start, end, period=period, adjust=adjust)
        except Exception as e:
            last_err = e
            continue
    if last_err:
        raise last_err
    return pd.DataFrame(columns=["date","open","high","low","close","volume","amount"])


# ========== 批量、多股 ==========
def get_stocks_batch(codes: Iterable[str], start: str, end: str, *,
                     period: str="daily", adjust: str="qfq",
                     return_long: bool=False,
                     providers_order: Optional[List[str]]=None) -> Dict[str, pd.DataFrame] | pd.DataFrame:
    out: Dict[str, pd.DataFrame] = {}
    for code in codes:
        try:
            df = get_stock_daily(code, start, end, period=period, adjust=adjust, providers_order=providers_order)
            if not df.empty:
                out[_norm_stock_code(code)] = df
        except Exception:
            continue
    if not return_long:
        return out
    if not out:
        return pd.DataFrame(columns=["code","date","open","high","low","close","volume","amount"])
    frames = []
    for c, df in out.items():
        t = df.copy()
        t.insert(0, "code", c)
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


def get_all_a_share_codes() -> pd.DataFrame:
    df = ak.stock_zh_a_spot_em()
    return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame(columns=["代码","名称"])


# ========== 指数（多源兜底） ==========
DEFAULT_INDEX_MAP = {
    "上证综指":"000001",
    "深证成指":"399001",
    "创业板指":"399006",
    "沪深300":"000300",
    "中证500":"000905",
    "中证全指":"000985",
    "上证50":"000016",
}

def _index_tushare_pull(code: str, start: str, end: str, *, period: str) -> pd.DataFrame:
    if ts is None: 
        raise RuntimeError("tushare 未安装")
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("未设置 TUSHARE_TOKEN")
    ts.set_token(token)
    pro = ts.pro_api()

    ts_code = _to_ts_code(code if code in ("000001","000016","000300","000905","000985") else code)  # 000001->000001.SH
    s, e = _to_ak_date(start), _to_ak_date(end)
    df = pro.index_daily(ts_code=ts_code, start_date=s, end_date=e)
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    df.rename(columns={"trade_date":"date","open":"open","high":"high","low":"low","close":"close","vol":"volume"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    # index_daily 没有 amount，这里补列为空
    if "amount" not in df.columns: df["amount"] = np.nan
    df = df[["date","open","high","low","close","volume","amount"]].sort_values("date").reset_index(drop=True)
    if period == "daily": return df
    tsdf = df.set_index("date").sort_index()
    rule = "W-FRI" if period=="weekly" else "M"
    parts = [
        tsdf["open"].resample(rule).first(),
        tsdf["high"].resample(rule).max(),
        tsdf["low"].resample(rule).min(),
        tsdf["close"].resample(rule).last(),
        tsdf["volume"].resample(rule).sum(min_count=1),
    ]
    out = (pd.concat(parts, axis=1).set_axis(["open","high","low","close","volume"], axis=1)
           .dropna(subset=["open","high","low","close"]).reset_index())
    out["amount"] = np.nan
    return out


def _index_akshare_pull(code: str, start: str, end: str, *, period: str) -> pd.DataFrame:
    s, e = _to_ak_date(start), _to_ak_date(end)
    def _primary():
        return ak.index_zh_a_hist(symbol=code, period=period, start_date=s, end_date=e)
    try:
        df = _with_retry(_primary, 5, 0.8)
    except Exception:
        df = None
    if isinstance(df, pd.DataFrame) and not df.empty:
        df = _rename_ohlc(df)
        if "date" in df.columns: df["date"] = pd.to_datetime(df["date"])
        for c in ["open","high","low","close","volume","amount"]:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.sort_values("date").reset_index(drop=True)

    # 备用：用 AkShare 的指数日线（新浪/东财的另一路）
    sym = ("sh" if code.startswith(("0","5")) else "sz") + code
    try:
        df2 = ak.stock_zh_index_daily(symbol=sym)
    except Exception:
        df2 = None
    if isinstance(df2, pd.DataFrame) and not df2.empty:
        df2 = _rename_ohlc(df2)
        if "date" in df2.columns: df2["date"] = pd.to_datetime(df2["date"])
        for c in ["open","high","low","close","volume","amount"]:
            if c in df2.columns: df2[c] = pd.to_numeric(df2[c], errors="coerce")
        df2 = df2.sort_values("date").reset_index(drop=True)
        if period == "daily": return df2
        tsdf = df2.set_index("date").sort_index()
        rule = "W-FRI" if period=="weekly" else "M"
        parts = [
            tsdf["open"].resample(rule).first(),
            tsdf["high"].resample(rule).max(),
            tsdf["low"].resample(rule).min(),
            tsdf["close"].resample(rule).last(),
        ]
        names = ["open","high","low","close"]
        if "volume" in tsdf.columns:
            parts.append(tsdf["volume"].resample(rule).sum(min_count=1)); names.append("volume")
        if "amount" in tsdf.columns:
            parts.append(tsdf["amount"].resample(rule).sum(min_count=1)); names.append("amount")
        out = (pd.concat(parts, axis=1).set_axis(names, axis=1)
               .dropna(subset=["open","high","low","close"]).reset_index())
        return out

    return pd.DataFrame()


def get_market_indices(start: str, end: str, *, period: str="daily",
                       indices: Optional[Dict[str,str]]=None,
                       return_long: bool=True,
                       providers_order: Optional[List[str]]=None) -> pd.DataFrame:
    id_map = indices or {
        "上证综指":"000001","深证成指":"399001","创业板指":"399006",
        "沪深300":"000300","中证500":"000905","中证全指":"000985","上证50":"000016",
    }
    if providers_order is None:
        providers_order = ["tushare","akshare"] if os.getenv("TUSHARE_TOKEN") else ["akshare","tushare"]

    frames: List[pd.DataFrame] = []
    for name, code in id_map.items():
        df = pd.DataFrame()
        for prov in providers_order:
            try:
                df = _index_tushare_pull(code, start, end, period=period) if prov=="tushare" else _index_akshare_pull(code, start, end, period=period)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    break
            except Exception:
                continue
        if df.empty: 
            continue
        df.insert(0, "index_code", code)
        df.insert(1, "index_name", name)
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["index_code","index_name","date","open","high","low","close","volume","amount"])
    long_df = pd.concat(frames, ignore_index=True)
    if return_long:
        return long_df
    wide = (long_df[["index_name","date","close"]]
            .pivot(index="date", columns="index_name", values="close")
            .sort_index().reset_index())
    return wide


# ========== 估值 / 财务 / 分红 ==========
def get_stock_valuation_hist(code: str) -> pd.DataFrame:
    """优先 Tushare daily_basic；失败则 AkShare（乐咕/百度）"""
    sym6 = _norm_stock_code(code)

    # Tushare
    try:
        if ts is not None and os.getenv("TUSHARE_TOKEN"):
            ts.set_token(os.getenv("TUSHARE_TOKEN"))
            pro = ts.pro_api()
            db = pro.daily_basic(ts_code=_to_ts_code(sym6), fields="trade_date,ts_code,pe,pe_ttm,pb,total_mv,circ_mv")
            if isinstance(db, pd.DataFrame) and not db.empty:
                db.rename(columns={"trade_date":"date"}, inplace=True)
                db["date"] = pd.to_datetime(db["date"])
                return (db[["date","pe","pe_ttm","pb","total_mv","circ_mv"]]
                        .sort_values("date").reset_index(drop=True))
    except Exception:
        pass

    # AkShare·乐咕/东财
    try:
        try:
            df = ak.stock_a_indicator_lg(symbol=sym6)
        except TypeError:
            df = ak.stock_a_indicator_lg(stock=sym6)
        if isinstance(df, pd.DataFrame) and not df.empty:
            m = {"trade_date":"date","日期":"date","pe":"pe","pe_ttm":"pe_ttm","pb":"pb",
                 "ps":"ps","ps_ttm":"ps_ttm","dv_ratio":"dv_ratio","dv_ttm":"dv_ttm",
                 "total_mv":"total_mv","circ_mv":"circ_mv"}
            df = df.rename(columns=m)
            if "date" in df.columns: df["date"] = pd.to_datetime(df["date"])
            keep = [c for c in ["date","pe","pe_ttm","pb","ps","ps_ttm","dv_ratio","dv_ttm","total_mv","circ_mv"] if c in df.columns]
            return df[keep].dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    except Exception:
        pass

    # AkShare·百度估值兜底
    try:
        df2 = ak.stock_zh_valuation_baidu(symbol=_mk_prefixed_code(sym6))
        if isinstance(df2, pd.DataFrame) and not df2.empty:
            mm = {"日期":"date","总市值":"total_mv","流通市值":"circ_mv",
                  "市盈率TTM":"pe_ttm","市盈率(静)":"pe","市净率":"pb","股息率":"dv_ratio"}
            df2 = df2.rename(columns=mm)
            if "date" in df2.columns: df2["date"] = pd.to_datetime(df2["date"])
            keep = [c for c in ["date","pe","pe_ttm","pb","dv_ratio","total_mv","circ_mv"] if c in df2.columns]
            return df2[keep].dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    except Exception:
        pass

    return pd.DataFrame(columns=["date","pe","pe_ttm","pb","ps","ps_ttm","dv_ratio","dv_ttm","total_mv","circ_mv"])


def get_stock_financial_abstract(code: str) -> pd.DataFrame:
    """
    财务摘要（年/季报宽表 → 规范长表）。
    AkShare 返回形态：列为 YYYYMMDD（如 '20250930'），行有“选项/指标”两列。
    本函数做：
      1) 调用 ak.stock_financial_abstract(symbol=...) 取宽表
      2) 识别日期列（^\d{8}$），melt 为长表：['date','metric','value']
      3) 选取关键指标并做英文映射：净利润/营收/ROE/... → net_profit/revenue/roe 等
      4) pivot 成：['date','net_profit','revenue','roe', ...]（数值为 float）
    返回：一定包含 'date' 列；无数据时返回空 DF（但列包含 'date'）
    """
    import re
    sym6 = _norm_stock_code(code)

    # 1) 拉宽表
    df = None
    for arg in ({"symbol": sym6}, (sym6,)):
        try:
            df = ak.stock_financial_abstract(**arg) if isinstance(arg, dict) else ak.stock_financial_abstract(*arg)
            if isinstance(df, pd.DataFrame) and not df.empty:
                break
        except Exception:
            df = None

    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=["date"])

    # 2) 找出所有日期列（YYYYMMDD）
    date_cols = [c for c in df.columns if isinstance(c, str) and re.fullmatch(r"\d{8}", c)]
    if not date_cols:
        # 极端情况：没有任何日期列，也返回至少含 'date'
        return pd.DataFrame(columns=["date"])

    # 3) 只保留“指标 + 日期列”，把“选项”保留用于后续 disambiguate
    df_wide = df[["选项", "指标"] + date_cols].copy()

    # 4) 宽 → 长
    long = df_wide.melt(id_vars=["选项", "指标"], value_vars=date_cols,
                        var_name="date_str", value_name="value")
    long["date"] = pd.to_datetime(long["date_str"], format="%Y%m%d", errors="coerce")
    long.drop(columns=["date_str"], inplace=True)

    # 数值化
    long["value"] = pd.to_numeric(long["value"], errors="coerce")

    # 5) 选择关键指标并映射英文名（模糊匹配，尽量兼容）
    #   常见：归母净利润/净利润, 营业总收入/营业收入, ROE(加权/摊薄), 经营活动现金流量净额 等
    metric_map = {
        "net_profit": ["归母净利润", "净利润", "归属于母公司股东的净利润"],
        "revenue": ["营业总收入", "营业收入", "主营业务收入"],
        "roe": ["净资产收益率", "ROE", "加权净资产收益率", "净资产收益率(加权)", "净资产收益率(摊薄)"],
        "ocf": ["经营活动产生的现金流量净额", "经营现金流量净额"],
        "gross_margin": ["毛利率"],
        "op_profit": ["营业利润"],
        "net_profit_margin": ["净利率", "销售净利率"],
    }

    def pick_metric_rows(name: str, keywords: list[str]) -> pd.DataFrame:
        mask = long["指标"].astype(str).apply(lambda x: any(k in x for k in keywords))
        sub = long[mask][["date", "value"]].copy()
        sub.rename(columns={"value": name}, inplace=True)
        return sub

    # 汇总各指标
    merged = None
    for eng, kws in metric_map.items():
        sub = pick_metric_rows(eng, kws)
        if sub.empty:
            continue
        merged = sub if merged is None else pd.merge(merged, sub, on="date", how="outer")

    if merged is None or merged.empty:
        # 至少要有 date 列
        return pd.DataFrame(columns=["date"])

    merged = merged.sort_values("date").reset_index(drop=True)
    # 确保所有数值列为 float
    for c in merged.columns:
        if c != "date":
            merged[c] = pd.to_numeric(merged[c], errors="coerce")

    return merged


def get_stock_bonus_dividend(code: str) -> pd.DataFrame:
    """分红派息/送转（AkShare·东财/Sina）"""
    sym6 = _norm_stock_code(code)
    try:
        df = ak.stock_fhps_em(symbol=sym6)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    except Exception:
        pass
    try:
        df2 = ak.stock_bonus(stock=sym6)
        return df2 if isinstance(df2, pd.DataFrame) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def merge_price_with_valuation(price_df: pd.DataFrame, val_df: pd.DataFrame) -> pd.DataFrame:
    if price_df is None or price_df.empty: return price_df
    if val_df is None or val_df.empty: return price_df
    return pd.merge(price_df, val_df, on="date", how="left")


# ========== 分钟级 / 资金面 / 日历 ==========
def get_stock_minute_bars(code: str, start: str, end: str, *, period: str="5", adjust: str="qfq") -> pd.DataFrame:
    sym = _mk_prefixed_code(_norm_stock_code(code))
    df = None
    try:
        df = ak.stock_zh_a_hist_min_em(symbol=sym, start_date=start, end_date=end, period=period, adjust=adjust)
    except Exception:
        df = None
    if not isinstance(df, pd.DataFrame) or df.empty:
        try:
            df = ak.stock_zh_a_minute(symbol=sym, period=period, adjust=adjust)
        except Exception:
            df = None
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=["date","open","high","low","close","volume","amount"])
    df = _rename_ohlc(df)
    if "date" in df.columns: df["date"] = pd.to_datetime(df["date"])
    for c in ["open","high","low","close","volume","amount"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open","high","low","close"]).sort_values("date").reset_index(drop=True)


def get_northbound_flow(start: str, end: str) -> pd.DataFrame:
    try:
        df = ak.stock_hsgt_fund_flow_summary()
        if not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame(columns=["date","north_net","sh_net","sz_net"])
        mm = {"日期":"date","沪股通净流入":"sh_net","深股通净流入":"sz_net","北向资金净流入":"north_net"}
        df = df.rename(columns=mm)
        if "date" in df.columns: df["date"] = pd.to_datetime(df["date"])
        for c in ["north_net","sh_net","sz_net"]:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        s, e = pd.to_datetime(start), pd.to_datetime(end)
        df = df[(df["date"] >= s) & (df["date"] <= e)].sort_values("date").reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame(columns=["date","north_net","sh_net","sz_net"])


def get_trade_calendar(start: str="1990-01-01", end: str="2100-12-31") -> pd.DataFrame:
    df = ak.tool_trade_date_hist_sina()
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=["date"])
    mm = {"trade_date":"date","日期":"date"}
    df = df.rename(columns=mm)
    if "date" in df.columns: df["date"] = pd.to_datetime(df["date"])
    s, e = pd.to_datetime(start), pd.to_datetime(end)
    return df[(df["date"] >= s) & (df["date"] <= e)].reset_index(drop=True)


# ========== 分析友好小助手 ==========
def as_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df
    out = df.copy()
    if "date" in out.columns:
        out = out.sort_values("date").set_index("date", drop=True)
    for c in ["open","high","low","close","volume","amount"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    return out


def pivot_close(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df is None or long_df.empty: return pd.DataFrame()
    return (long_df[["code","date","close"]].dropna()
            .pivot(index="date", columns="code", values="close")
            .sort_index())


def assert_ready_for_ta(df: pd.DataFrame):
    need = {"open","high","low","close"}
    miss = need - set(df.columns)
    assert not miss, f"缺少列：{miss}"
    if "date" in df.columns:
        assert df["date"].is_monotonic_increasing, "日期未升序"
    sample = df.sample(min(len(df), 10), random_state=0)[list(need)]
    assert sample.applymap(lambda x: isinstance(x, (int,float)) and x == x).all().all(), "存在非数值或 NaN"


# ========== CLI ==========
if __name__ == "__main__":
    import argparse
    import datetime as dt

    p = argparse.ArgumentParser(description="A股数据获取（多提供商 + 自动兜底）")
    p.add_argument("--code", help="单只股票，如 600519 或 600519.SH")
    p.add_argument("--codes", help="多只股票，逗号分隔，如 600519,000001.SZ")
    p.add_argument("--market", action="store_true", help="输出常用大盘指数（长表）")
    p.add_argument("--minute", action="store_true", help="输出单只股票的分钟K（配合 --code）")
    p.add_argument("--period", default="daily", choices=["daily","weekly","monthly"], help="日/周/月（分钟K独立）")
    p.add_argument("--mperiod", default="5", choices=["1","5","15","30","60"], help="分钟周期（配合 --minute）")
    p.add_argument("--adjust", default="qfq", choices=["","qfq","hfq"], help="复权方式（价格）")
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default=dt.date.today().strftime("%Y-%m-%d"))
    p.add_argument("--latest", type=int, default=10, help="仅显示最近N行（默认10）")
    p.add_argument("--provider-order", default="", help="提供商优先级，逗号分隔，如 tushare,akshare,baostock")

    p.add_argument("--with-valuation", action="store_true", help="同时获取估值并与价格合并（配合 --code）")
    p.add_argument("--with-financial", action="store_true", help="输出财务摘要（配合 --code）")
    p.add_argument("--with-bonus", action="store_true", help="输出分红派息（配合 --code）")
    p.add_argument("--northbound", action="store_true", help="输出北向资金净流入（按区间）")

    args = p.parse_args()

    prov_order = [s.strip() for s in args.provider_order.split(",") if s.strip()] or None

    if args.code and not args.minute:
        px = get_stock_daily(args.code, args.start, args.end, period=args.period, adjust=args.adjust, providers_order=prov_order)
        if args.with_valuation:
            val = get_stock_valuation_hist(args.code)
            px = merge_price_with_valuation(px, val)
        print(px.tail(args.latest))
        if args.with_financial:
            fa = get_stock_financial_abstract(args.code)
            print("\n[财务摘要（截断显示）]")
            print(fa.tail(8))
        if args.with_bonus:
            bo = get_stock_bonus_dividend(args.code)
            print("\n[分红派息（截断显示）]")
            print(bo.tail(8))

    if args.code and args.minute:
        mk = get_stock_minute_bars(args.code, args.start, args.end, period=args.mperiod, adjust=args.adjust)
        print(mk.tail(args.latest))

    if args.codes:
        cs = [c.strip() for c in args.codes.split(",") if c.strip()]
        df_long = get_stocks_batch(cs, args.start, args.end, period=args.period, adjust=args.adjust, return_long=True, providers_order=prov_order)
        print(df_long.tail(args.latest))

    if args.market:
        idx = get_market_indices(args.start, args.end, period=args.period, return_long=True, providers_order=prov_order)
        print(idx.tail(args.latest))

    if args.northbound:
        nb = get_northbound_flow(args.start, args.end)
        print(nb.tail(args.latest))

    # 交易日历按需自行调用
    # cal = get_trade_calendar(args.start, args.end)
    # print(cal.tail(10))
