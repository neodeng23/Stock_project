# -*- coding: UTF-8 -*-
import os
import requests
import json
import xlrd
import xlutils
from xlutils.copy import copy
import sys


def get_device_spec(hostname):
    url = "http://abc.def.com/url/s"
    pa = "_type:server,hostname:" + hostname
    param = {"q": pa}
    respone = requests.request("GET", url, params=param)
    res = json.loads(respone.text)
    server_detail = res["result"]
    return server_detail


def write_excel_data(file_name, sheet_num, row, col, value_data):
    rbook = xlrd.open_workbook(file_name)
    wbook = copy(rbook)
    w_sheet = wbook.get_sheet(sheet_num)
    w_sheet.write(row, col, value_data)
    wbook.save(file_name)


fileName = "11.xlsx"
data = xlrd.open_workbook(fileName)
sheet01 = data.sheets()[0]
sheet01_nrows = sheet01.nrows
sheet01_row = 1

for sheet01_row in range(1, sheet01_nrows):
    hostname = sheet01.cell(sheet01_row, 0).value
    server = get_device_spec(hostname)
    try:
        write_excel_data(fileName, 0, sheet01_row, 1, server[0]["manufacturer"])
        write_excel_data(fileName, 0, sheet01_row, 2, server[0]["idc"])
        write_excel_data(fileName, 0, sheet01_row, 3, server[0]["sn"])
        write_excel_data(fileName, 0, sheet01_row, 4, server[0]["rack_location"])
        write_excel_data(fileName, 0, sheet01_row, 5, server[0]["flavor_disk"])
        write_excel_data(fileName, 0, sheet01_row, 6, server[0]["maintain_enddate"])

    except Exception as e:
        pass
    continue


