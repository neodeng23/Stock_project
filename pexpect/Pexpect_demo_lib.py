import pexpect
from pexpect.pxssh import pxssh
import sys


# 远程登录主机并执行命令
def login():
    child = pexpect.spawn('ssh ginvip@172.17.2.117')
    child.logfile = sys.stdout
    child.expect('password')
    child.sendline('ginvip')
    child.expect('ginvip')
    child.sendline('ls /')
    child.expect('ginvip')
    child.sendline('exit')


# 将日志写入文件
def write():
    child = pexpect.spawn('ssh ginvip@172.17.2.117')
    fout = file('log.txt', 'w')
    child.logfile = fout
    child.expect('password')
    child.sendline('ginvip')
    child.expect('ginvip')
    child.sendline('ls /')
    child.expect('ginvip')
    child.sendline('exit')


def sshlogin():
    hostname = '172.17.2.117'
    user = 'ginvip'
    password = 'ginvip'
    s = pxssh()
    s.login(hostname, user, password)
    s.sendline('ip addr')
    s.prompt()  # 匹配命令提示符
    print(s.before)  # 查看命令执行结果
    s.logout()
