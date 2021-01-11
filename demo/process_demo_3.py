import multiprocessing
import time


def func(arg):
    pname = multiprocessing.current_process().name
    pid = multiprocessing.current_process().pid
    print("当前进程ID=%d,name=%s" % (pid, pname))
    while True:
        print(arg)
        time.sleep(1)
        pass


if __name__ == "__main__":
    p = multiprocessing.Process(target=func, args=("hello",))
    p = multiprocessing.Process(target=func, name="劳资的队伍", args=("hello",))
    p.daemon = True  # 设为【守护进程】（随主进程的结束而结束）
    p.start()
    while True:
        print("子进程是否活着？", p.is_alive())
        time.sleep(1)
        pass
