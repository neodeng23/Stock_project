import xlrd
import getpass
from xlrd import open_workbook
from xlutils.copy import copy

User_name = getpass.getuser()


def read_excel(filename):
    rexcel = open_workbook('/Users/' + User_name + '/Desktop/' + filename, formatting_info=True)   # 保留原有样式
    sheet_num = len(rexcel.sheets())

    # 用 xlutils 提供的copy方法将 xlrd 的对象转化为 xlwt 的对象
    excel = copy(rexcel)

    # 用 xlwt 对象的方法获得要操作的 sheet
    table = excel.get_sheet(0)

    for i in range(len(sheet_num)):

        sheet = excel.get_sheet(i)

        print('\n\n第' + str(i + 1) + '个sheet: ' + sheet.title + '->>>')

        for r in range(1, sheet.max_row + 1):
            if r == 1:
                print('\n' + ''.join(
                    [str(sheet.cell(row=r, column=c).value).ljust(17) for c in range(1, sheet.max_column + 1)]))
            else:
                print(
                    ''.join([str(sheet.cell(row=r, column=c).value).ljust(20) for c in range(1, sheet.max_column + 1)]))
