from VREP import *
from PointCloud import *
from Utils import *
from YOLO.YOLO import *
from multiprocessing import Queue


def main():
    """
    main function used to control sub-processes and queues
    """
    # finish event for all threads and processes
    finish = Event()

    # create queue dictionaries
    #vrep_queues = {"input": RQueue(finish, method="wait"), "output": RQueue(finish, method="wait")}
    # the queues are setup as though they are a loop, starting and ending at vrep
    vrep_queues = {"input": RQueue(finish), "output": RQueue(finish)}
    yolo_queues = {"input": vrep_queues["output"], "output": RQueue(finish)}
    pcloud_queues = {"input": yolo_queues["output"], "output": vrep_queues["input"]}
    #pcloud_queues = {"input": vrep_queues["output"], "output": vrep_queues["input"]}

    # create sub processes and list them
    vrep = SubProcessClass(target=VREP, finish=finish, args=(vrep_queues,))
    point_cloud = SubProcessClass(target=PointCloud, finish=finish, args=(pcloud_queues,))
    yolo = SubProcessClass(target=YOLO, finish=finish, args=(yolo_queues,))
    processes = [vrep, point_cloud, yolo]
    [p.start() for p in processes]

    # create main loop
    while not finish.is_set():
        time.sleep(1/100)

    print("main loop broken")
    cv2.destroyAllWindows()
    time.sleep(2)

    # close processes if not already closed
    [(print(f"closing {p}"), p.terminate()) if p.is_alive() else print(f"{p} already closed") for p in processes]


if __name__ == "__main__":
    main()
