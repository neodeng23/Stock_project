import multiprocessing
import threading
import time

data = 0


def sendEvent(event):
    global data
    for i in range(5):
        data += 1
        event.set()
        print("事件已发送*", data)
        time.sleep(1)
    pass


def handleEvent(event):
    global data
    for i in range(5):
        data += 1
        event.wait()
        print("事件已处理*", data)
        event.clear()
    pass


if __name__ == "__main__":
    # 进程通信
    # event = multiprocessing.Event()
    # p1 = multiprocessing.Process(target=sendEvent, args=(event,))
    # p2 = multiprocessing.Process(target=handleEvent, args=(event,))
    # p1.start()
    # p2.start()
    # p2.join()

    # 线程通信
    event = threading.Event()
    threading.Thread(target=sendEvent, args=(event,)).start()
    threading.Thread(target=handleEvent, args=(event,)).start()

    print("data=", data)