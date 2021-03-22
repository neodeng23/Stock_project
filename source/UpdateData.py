# code:utf-8
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from stock_hwnd import *
import sys
import time
import tushare as ts

pro = ts.pro_api('af62c3914f45eeb66261db1c4624d48d4f37fc58ee56ac4a204d27f4')
stock_code = ['000001', '000002', '000023']


class UpdateData(QThread):
    """更新数据类"""
    df = realtime_quotes(stock_code)
    update_date = pyqtSignal(type(df))  # pyqt5 支持python3的str，没有Qstring

    def run(self):
        while True:
            df = realtime_quotes(stock_code)
            print(df)
            self.update_date.emit(df)
            time.sleep(1)
