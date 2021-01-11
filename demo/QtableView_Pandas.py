import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication, QTableView
from PyQt5.QtCore import QAbstractTableModel, Qt
from stock_hwnd import *
import tushare as ts
from time import sleep
import datetime


# df = pd.DataFrame({'a': ['Mary', 'Jim', 'John'],
#                    'b': [100, 200, 300],
#                    'c': ['a', 'b', 'c']})

df = realtime_quotes("000001")[["name", "pre_close", "price"]]


class pandasModel(QAbstractTableModel):

    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parnet=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None


if __name__ == '__main__':
    pro = ts.pro_api('af62c3914f45eeb66261db1c4624d48d4f37fc58ee56ac4a204d27f4')
    app = QApplication(sys.argv)
    model = pandasModel(df)
    view = QTableView()
    view.setModel(model)
    view.resize(800, 600)
    view.show()
    sys.exit(app.exec_())