# -*- coding:utf-8 -*-
import win32gui
import win32con
import method
import time
import numpy as np
import pandas as pd
import control
import method
from stock_hwnd import *

wdname = "网上股票交易系统5.0"
stock_code = '002507'


def buy_stock(code):
    hwnd = win32gui.FindWindow(None, wdname)
    win32gui.SetForegroundWindow(hwnd)
    control.key_Press(112)

    hwnd_buy_list = hwnd_buy(wdname)
    method.sendEditText(hwnd_buy_list[0], code)


if __name__ == "__main__":
    buy_stock(stock_code)
