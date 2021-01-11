import win32gui
import win32con
import win32api
from win32gui import *


def get_child_windows(wdname):
    """
    获得parent的所有子窗口句柄
    返回子窗口句柄列表
    """
    parent = win32gui.FindWindow(None, wdname)
    if not parent:
        return
    hwndChildList = []
    win32gui.EnumChildWindows(parent, lambda hwnd, param: param.append(hwnd),  hwndChildList)
    return hwndChildList


def get_child_windows_hwnd(parent):
    """
    获得parent的所有子窗口句柄
    返回子窗口句柄列表
    """
    if not parent:
        return
    hwndChildList = []
    win32gui.EnumChildWindows(parent, lambda hwnd, param: param.append(hwnd),  hwndChildList)
    return hwndChildList


def reset_window_pos(targetTitle):
    """
    Python遍历窗口并设置窗口位置的方法
    """
    hWndList = []
    win32gui.EnumWindows(lambda hWnd, param: param.append(hWnd), hWndList)
    for hwnd in hWndList:
        clsname = win32gui.GetClassName(hwnd)
        title = win32gui.GetWindowText(hwnd)
        if title.find(targetTitle) >= 0:  # 调整目标窗口到坐标(600,300),大小设置为(600,600)
            print("找到"+targetTitle)
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 600, 300, 1000, 1000, win32con.SWP_SHOWWINDOW)


def getEditText(hwnd):
    """
    获取文本
    :param hwnd:
    :return:
    """
    bufLen = win32gui.SendMessage(hwnd, win32con.WM_GETTEXTLENGTH, 0, 0) + 1
    # print("bufLen:", bufLen)
    buffer = win32gui.PyMakeBuffer(bufLen)
    win32gui.SendMessage(hwnd, win32con.WM_GETTEXT, bufLen, buffer)
    address, length = PyGetBufferAddressAndLen(buffer)
    text = PyGetString(address, length)
    return text


def sendEditText(hwnd, text):
    text_ob = win32gui.FindWindowEx(hwnd, None, "Edit", None)
    win32api.SendMessage(text_ob, win32con.WM_SETTEXT, None, text)

# 输出当前活动窗体句柄
def print_GetForegroundWindow():
    hwnd_active = win32gui.GetForegroundWindow()
    print("hwnd_active hwnd:", hwnd_active)
    print("hwnd_active text:", win32gui.GetWindowText(hwnd_active))
