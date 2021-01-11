import winreg
import re
import serial.tools.list_ports


def get_serial():
    string = r'SYSTEM\CurrentControlSet\Control\COM Name Arbiter\Devices'   # 注册表路径
    handle = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, string, 0, (winreg.KEY_WOW64_64KEY + winreg.KEY_READ))
    com_list = []
    com_dict = {}

    port_list = list(serial.tools.list_ports.comports())
    if len(port_list) == 0:
        print('无可用串口')
    else:
        for i in range(0, len(port_list)):
            com_name = port_list[i].name
            com_list.append(com_name)

    for x in com_list:
        location, _type = winreg.QueryValueEx(handle, x)
        com_dict[x] = location
    return com_dict


def serial_test(com_dict):
    if len(com_dict) == 0:
        print('一个串口都没有，请检查串口连接！！！！！')
    else:
        num = 0
        for key in com_dict:
            name = com_dict[key]
            if len(re.findall("a50285b", name)) > 0:
                num = 1
                print("已找到" + name)
    if num == 1:
        print("已找到FIXTURE")
        return "ok"
    else:
        print("请检查串口！！！！！")
        return "fail"


def get_serial_FIXTURE(com_dict):
    if len(com_dict) == 0:
        print('一个串口都没有，请检查串口连接！！！！！')
    else:
        for key in com_dict:
            name = com_dict[key]
            if len(re.findall("a50285b", name)) > 0:
                return key

dict = get_serial()
print(dict)