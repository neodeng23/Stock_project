# -*- coding:utf-8 -*-
import win32gui
import win32con
import method
import time
import tushare as ts
import numpy as np
import pandas as pd

# pro = ts.pro_api('af62c3914f45eeb66261db1c4624d48d4f37fc58ee56ac4a204d27f4')  # 接口初始化
# stock_code = '000001'
ID_1 = "0000E900"  # 1级childhwnd
ID_2 = "0000E901"  # 2级childhwnd
ID_code = "00000408"  # 证券代码
ID_name = "0000040C"  # 证券名称（读）
ID_buyInPrice = "00000409"  # 买(卖)入价格
ID_accNum = "000003FA"  # 可买入股（读）
ID_buyNUm = "0000040A"  # 买（卖）入数量
ID_buyButton = "000003EE"  # 买（卖）入键

"""
需要配合华泰的界面切换，当处于买入界面时，此方法调出的hwnd是买入的，
当换成卖出界面时，则调出卖出的hwnd
"""


def hwnd_buy(wdname):
    main_hwnd = win32gui.FindWindow(None, wdname)
    hwnd_01 = win32gui.GetDlgItem(main_hwnd, int(ID_1, 16))
    hwnd_02 = win32gui.GetDlgItem(hwnd_01, int(ID_2, 16))
    hwnd_buy_list = [
        win32gui.GetDlgItem(hwnd_02, int(ID_code, 16)),
        win32gui.GetDlgItem(hwnd_02, int(ID_name, 16)),
        win32gui.GetDlgItem(hwnd_02, int(ID_buyInPrice, 16)),
        win32gui.GetDlgItem(hwnd_02, int(ID_accNum, 16)),
        win32gui.GetDlgItem(hwnd_02, int(ID_buyNUm, 16)),
        win32gui.GetDlgItem(hwnd_02, int(ID_buyButton, 16))]
    return hwnd_buy_list


# 证券代码 # 证券名称（不可填） # 买出价格 # 可买股（不可填） # 买出数量 # 买入键


def realtime_quotes(stock_code):
    df = ts.get_realtime_quotes(stock_code)
    return df
