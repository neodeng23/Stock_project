# -*- coding: utf-8 -*-
import time
import re
import winreg
import serial
import serial.tools.list_ports

timeout = "3000"
endsymbol = '0D0A'
baud_Rate = "115200"


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
    global num
    if len(com_dict) == 0:
        print('一个串口都没有，请检查串口连接！！！！！')
    else:
        num = 0
        for key in com_dict:
            name = com_dict[key]
            if len(re.findall("fixture", name)) > 0:
                num = num + 1
                print("已找到控制板")
            if len(re.findall("feasa", name)) > 0:
                num = num + 1
                print("已找到feasa")
            if len(re.findall("a50285bi", name)) > 0:
                num = num + 1
                print("已找到DUT")
    if num == 3:
        print("已找到FEASA，FIXTURE，A50285BI")
    else:
        print("请检查串口！！！！！")


def get_serial_FIXTURE(com_dict):
    if len(com_dict) == 0:
        print('一个串口都没有，请检查串口连接！！！！！')
    else:
        for key in com_dict:
            name = com_dict[key]
            if len(re.findall("fixture", name)) > 0:
                return key


def monitor(PortName):
    ser = serial.Serial(PortName, 115200)
    ser.flushInput()
    print('请双启动。。。。。。。')
    while True:
        count = ser.inWaiting()  # 获取串口缓冲区数据
        if count != 0:
            recv = ser.read(ser.in_waiting)  # 读出串口数据
            print(recv)  # 打印一下子
            if recv == b'Start\r\n':
                ser.close()
                break
        time.sleep(0.1)  # 延时0.1秒，免得CPU出问题
    return "ok"


def CMD_test(PortName):
    cmd_flag = 0
    for cmd in [
                "ActionButtonDown",
                "ActionButtonUp",
                "ActionBudUp",
                "ActionCoverIn",
                "ActionCoverOut",
                "ActionReset",
                "TurnChargeVbusVoltage",
                "TurnLLink",
                "BatteryOn",
                "BatteryOff"
                ]:
        res = communiate(PortName, cmd, baud_Rate, timeout, endsymbol)
        if len(re.findall("OK", res)) > 0:
            print(cmd + " 已收到 " + res)
            cmd_flag = cmd_flag + 1
        else:
            print(cmd + " 执行失败 " + res)
        time.sleep(1)
    if cmd_flag == 10:
        return "ok"
    else:
        return "fail"


def communiate(PortName, cmd, BaudRate, timeout, endsymbol):
    cmd += '\r\n'
    ser = serial.Serial(PortName, int(BaudRate), timeout=float(timeout))
    ser.flush()

    trans_cmd = cmd.encode()
    ser.write(trans_cmd)

    # timeout==0 表示只写不读
    if float(timeout) == 0:
        ser.flush()
        ser.close()
        return ''

    _end = endsymbol
    if _end == '0D0A':
        _end = '\r\n'

    reports = ''
    tickbegin = time.time()
    while True:
        tickend = time.time()
        if (tickend - tickbegin) >= float(timeout):
            ser.flush()
            ser.close()
            break

        time.sleep(0.001)
        reports += ser.read(ser.inWaiting()).decode()

        if not endsymbol == 'no endSymble':
            if endsymbol in reports:
                ser.flush()
                ser.close()
                return reports
                break

            if _end in reports:
                ser.flush()
                ser.close()
                return reports
                break


def main():
    try:
        flag = 0
        com_dict = get_serial()
        PortName = get_serial_FIXTURE(com_dict)

        if serial_test(com_dict) == "ok":
            flag = flag + 1
            print("成功，找到控制板串口")
        else:
            print("失败，未找到所有串口")

        if monitor(PortName) == "ok":
            flag = flag + 1
            print("成功，双启动收到OK")
        else:
            print("失败，双启动未收到OK")

        time.sleep(3)

        if CMD_test(PortName) == "ok":
            flag = flag + 1
            print("成功，CMD已执行")
        else:
            print("失败，CMD未能执行")

        if flag == 3:
            print("机台验证完成！全部PASS")
        else:
            print("机台验证失败！！！！！！！！！请检查log")

    finally:
        input('Press any key to quit program.')


if __name__ == '__main__':
    main()
