# -*- coding: utf-8 -*-
import os
from multiprocessing import Process
import time


def fun(name):
    print("2 子进程信息： pid=%s, ppid=%s" % (os.getpid(), os.getppid()))
    print("hello " + name)


def test():
    print('ssss')


if __name__ == "__main__":
    print("主进程信息： pid=%s, ppid=%s" % (os.getpid(), os.getppid()))

    ps = Process(target=fun, args=('jingsanpang', ))

    print("111 ##### ps pid: " + str(ps.pid) + ", ident:" + str(ps.ident))

    print("1进程信息： pid=%s, ppid=%s" % (os.getpid(), os.getppid()))

    print(ps.is_alive())  # 启动之前 is_alive为False(系统未创建)
    ps.start()
    print(ps.is_alive())  # 启动之后，is_alive为True(系统已创建)

    print("222 #### ps pid: " + str(ps.pid) + ", ident:" + str(ps.ident))

    print("2进程信息： pid=%s, ppid=%s" % (os.getpid(), os.getppid()))
    ps.join()    # 等待子进程完成任务   类似于os.wait()

    print(ps.is_alive())
    print("3进程信息： pid=%s, ppid=%s" % (os.getpid(), os.getppid()))
    ps.terminate()  # 终断进程
    print("4进程信息： pid=%s, ppid=%s" % (os.getpid(), os.getppid()))
