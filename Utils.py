from multiprocessing import queues, Event, get_context, Process
import time
import sys
import cv2
import traceback
import numpy as np
import keyboard


class RQueue(queues.Queue):
    def __init__(self, finish,
                 method="recent",
                 ctx=get_context("spawn"),
                 maxsize=1):
        """
        This class is used to override queue in order to
        send the most recent data
        """
        queues.Queue.__init__(self, maxsize=maxsize, ctx=ctx)
        self.finish = finish
        self.requested = Event()
        self.method = method

    def retrieve(self, method=None):
        # if no method is detailed, used method in init
        if method is None:
            method = self.method

        if method == "recent":
            start = time.time()
            # sets the request, waits for message while checking finish status
            while not self.finish.is_set():
                # if not set, set request
                if not self.requested.is_set():
                    self.requested.set() # needs to be changed
                # if queue not empty, get contents and return
                if not self.empty():
                    self.requested.clear()
                    return self.get()
                time.sleep(1/100)

        elif method == "wait":
            # waits until queue is not empty while checking finish status
            start = time.time()
            while not self.finish.is_set():
                if not self.empty():
                    return self.get()
                time.sleep(1 / 100)

        elif method == "no_wait":
            if not self.empty():
                return self.get()
        else:
            raise NameError("incorrect method")

    def send(self, value, method=None):
        # if no method is detailed, used method in init
        if method is None:
            method = self.method

        if method == "recent":
            # only places value in queue if requested
            if self.requested.is_set():
                self.put(value)

        elif method == "wait":
            # waits until queue is not full to put value
            start = time.time()
            while not self.finish.is_set():
                if not self.full():
                    self.put(value)
                    break
                time.sleep(1/100)

        elif method == "no_wait":
            if not self.full():
                self.put(value)
        else:
            raise NameError("incorrect method")


def SubProcessClass(target, finish, args, method="fork"):
    """
    :param target: class to be targeted to be subprocess
    :param finish: finish event used to break loop
    :param queue_dict: dictionary of queues used for communication
    :param method: start method with default using spawn which is fastest to launch but uses more memory
    :return: class object which overrides the process class
    """
    ctx = get_context(method=method)

    class SubProcessOverride(ctx.Process):
        def __init__(self, target, finish, args):
            """
            overrides the Process class to create a seperate process containing this class only-
            input and output can be found in the subprocess and main process-
            """
            ctx.Process.__init__(self)
            self.finish = finish
            self.target = target
            self.args = args

        def run(self):
            """
            initialise used instead of init self because...
            memory from init self does not transfer to new class.

            Class requires three main methods;
                -init
                -main_loop
                -cleanup
            """
            try:
                # init target class
                self.target = self.target(*self.args)
            except Exception:
                traceback.print_exc()
                if not self.finish.is_set():
                    self.finish.set()
                return

            try:
                # loop targets main loop, while checking finish status
                while not self.finish.is_set():
                    # returns none, unless a break is called from within the class
                    if self.target.main_loop() == "break":
                        break
            # if exception occurs, raise error and print
            except Exception as e:
                traceback.print_exc()
            finally:
                # if code breaks, and finish has not been called, set the event
                if not self.finish.is_set():
                    self.finish.set()
                # call class cleanup code
                self.target.cleanup()

    return SubProcessOverride(target, finish, args)


class NamedWindow(object):
    def __init__(self, name, place=(0, 0), size=(300, 300), conversions=None):
        """
        used to simplify creating windows, customizing sizes and placing
        """
        self.name = name
        # create window
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(name, place[0], place[1])
        cv2.resizeWindow(name, size[0], size[1])
        self.conversions = conversions

    def show(self, img, max_value=None):
        """
        different methods can convert the image
        """
        if self.conversions is None:
            cv2.imshow(self.name, img)
        elif self.conversions == "heat_map":
            img = 255 * (img / max_value)
            # make sure img does not exceed max uint8 value
            img[np.where(img > 255)] = 255
            cv2.imshow(self.name,
                       cv2.applyColorMap(cv2.convertScaleAbs(img, 1), cv2.COLORMAP_JET))


def PrintFunctionTime(name, T, r=3):
    dur = time.time() - T
    if dur == 0:
        print(f"{name} takes: {round( dur * 1000, r)}ms, FPS: inf")
    else:
        print(f"{name} takes: {round( dur * 1000, r)}ms, FPS: {round(1 / dur, r)}")


class Clicker(object):
    def __init__(self, key, activated):
        self.key = key
        self.just_pressed = time.time()
        self.relax = 1/10
        self.activated = activated

    def click(self):
        # ensure that multiple signals are not sent for the same press
        if keyboard.is_pressed(self.key):
            if time.time() - self.just_pressed < self.relax:
                return self.activated

            elif self.activated:
                self.just_pressed = time.time()
                self.activated = False

            elif not self.activated:
                self.just_pressed = time.time()
                self.activated = True

        return self.activated
