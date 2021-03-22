import sys
import time
import stock_hwnd
import tushare as ts
from UpdateData import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *


pro = ts.pro_api('af62c3914f45eeb66261db1c4624d48d4f37fc58ee56ac4a204d27f4')
stock_code = ['000001', '000002', '000023']


class TableView(QWidget):
    def __init__(self, arg=None):
        super(TableView, self).__init__(arg)
        #self.setWindowTitle("股票行情")
        #self.resize(1600, 300)
        df = stock_hwnd.realtime_quotes(stock_code)
        self.get_update_data(df)
        self.tableview.setModel(self.model)
        layout = QVBoxLayout()
        layout.addWidget(self.tableview)
        self.setLayout(layout)

    def get_update_data(self, df):
        print(df)
        row_num, column_num = df.shape
        column_name = (list(df.columns))
        self.model = QStandardItemModel(row_num, column_num)
        self.model.setHorizontalHeaderLabels(column_name)  # 设置列名称
        self.tableview = QTableView()
        self.tableview.setModel(self.model)
        for x in range(0, row_num):
            for y in range(0, column_num):
                item = QStandardItem(df.iloc[x, y])
                self.model.setItem(x, y, item)

    def update_item_data(self, df):
        """更新内容"""
        row_num, column_num = df.shape
        for x in range(0, row_num):
            for y in range(0, column_num):
                item = QStandardItem(df.iloc[x, y])
                self.model.setItem(x, y, item)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myTable = TableView()

    # 启动更新线程
    update_data_thread = UpdateData()
    update_data_thread.update_date.connect(myTable.update_item_data)  # 链接信号
    update_data_thread.start()

    myTable.show()
    sys.exit(app.exec_())
