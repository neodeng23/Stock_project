from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
from WPS import Ui_Form
import sys


class MyMainForm(QMainWindow, Ui_Form):
    def __init__(self, parent=None):
        super(MyMainForm, self).__init__(parent)
        self.setupUi(self)

    def refresh_hold(self):
        pass

    def showtime(self):
        datetime = QTime.currentTime()
        text = datetime.toString()
        self.label_time.setText(text)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MyMainForm()

    # 建立线程刷新表格控件
    #data_widget = mainWindow.tableWidget
    #update_data_thread = UpdateData()
    #update_data_thread.update_date.connect(data_widget.update_item_data)  # 链接信号
    #update_data_thread.start()

    mainWindow.show()
    sys.exit(app.exec_())
