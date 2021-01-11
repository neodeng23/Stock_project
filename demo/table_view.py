import sys
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.uic import loadUi


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        loadUi('qtdesigner.ui', self)
        self.pushButton.clicked.connect(self.say)
        self.showData()

    def say(self):
        self.label.setText("哈哈哈")
        print("哈哈哈")

    def showData(self):
        # 准备数据模型
        self.sm = QtGui.QStandardItemModel()

        # 设置数据头栏名称
        self.sm.setHorizontalHeaderItem(0, QtGui.QStandardItem("Name"))
        self.sm.setHorizontalHeaderItem(1, QtGui.QStandardItem("NO."))

        # 设置数据条目
        self.sm.setItem(0, 0, QtGui.QStandardItem("张三"))
        self.sm.setItem(0, 1, QtGui.QStandardItem("20120202"))

        self.sm.setItem(1, 0, QtGui.QStandardItem("李四"))
        self.sm.setItem(1, 1, QtGui.QStandardItem("20120203000000000000000"))

        self.sm.setItem(2, 0, QtGui.QStandardItem("李四"))
        self.sm.setItem(2, 1, QtGui.QStandardItem("20120203000000000000000"))

        self.sm.setItem(3, 0, QtGui.QStandardItem("李四"))
        self.sm.setItem(3, 1, QtGui.QStandardItem("20120203000000000000000"))

        self.sm.setItem(4, 0, QtGui.QStandardItem("李四"))
        self.sm.setItem(4, 1, QtGui.QStandardItem("20120203000000000000000"))

        self.sm.setItem(5, 0, QtGui.QStandardItem("李四"))
        self.sm.setItem(5, 1, QtGui.QStandardItem("20120203000000000000000"))

        # 设置条目颜色和字体
        self.sm.item(0, 0).setForeground(QtGui.QBrush(QtGui.QColor(255, 0, 0)))
        self.sm.item(0, 0).setFont(QtGui.QFont("Times", 10, QtGui.QFont.Black))

        self.sm.item(3, 1).setBackground(QtGui.QBrush(QtGui.QColor(255, 255, 0)))

        # 按照编号排序
        self.sm.sort(1, QtCore.Qt.DescendingOrder)

        # 将数据模型绑定到QTableView
        self.tableView.setModel(self.sm)

        # QTableView
        self.tableView.setColumnWidth(0, 100)
        self.tableView.setColumnWidth(1, 200)


app = QApplication(sys.argv)
w = MainWindow()
w.show()
sys.exit(app.exec())