import sim
from threading import Thread
import keyboard
from DepthMapper import *
from Utils import *


class VREP(object):
    def __init__(self, queue_dict, sync=True):
        """
        this class loads coppeliasim and is responsible for communication between the sim and python...
        api functions can be found https://www.coppeliarobotics.com/helpFiles/en/remoteApiFunctionsPython.htm
        """
        # open queues
        self.queues = queue_dict

        # close all connection that are remaining
        sim.simxFinish(-1)
        self.clientID = -1
        attempt_num = 0
        while self.clientID == -1:
            print("attempting to connect to VREP...")
            self.clientID = sim.simxStart('127.0.0.1', 19999,
                                          True, True, 5000, 5)
            attempt_num += 1
            if attempt_num >= 3:
                print("could not connect to vrep")
                return
        print("successful connection!")

        self.sync = sync
        if self.sync:
            # set the simulation to synchronise with api
            sim.simxSynchronous(self.clientID, True)

        # get coppeliasim object handles
        self.motor = {'left': self.getHandle("Pioneer_p3dx_leftMotor"),
                      'right': self.getHandle("Pioneer_p3dx_rightMotor")}
        self.camHandle = self.getHandle("camera")
        #self.gripperHandle = {"left": self.getHandle("left_joint"),
        #                      "right": self.getHandle("right_joint")}

        # init camera data stream with rgb and depth
        sim.simxGetVisionSensorImage(self.clientID, self.camHandle, 0, sim.simx_opmode_streaming)
        sim.simxGetVisionSensorDepthBuffer(self.clientID, self.camHandle, sim.simx_opmode_streaming)

        # init position data stream with cartesian position and euler orientation
        sim.simxGetObjectOrientation(self.clientID, self.camHandle, -1, sim.simx_opmode_streaming)
        sim.simxGetObjectPosition(self.clientID, self.camHandle, -1, sim.simx_opmode_streaming)

        # setting motor speed
        self.speed = 1
        self.lr = 0.3
        self.images = None

        # remote start simulation if not already
        sim.simxStartSimulation(self.clientID, sim.simx_opmode_oneshot)

        # used for making keyboard switches
        self.keyboard_key = Clicker("c", activated=True)
        self.keyboard_controlled = self.keyboard_key.activated
        self.gripper_key = Clicker("g", activated=False)
        self.gripper_activated = self.gripper_key.activated

    def main_loop(self):
        """
        this is looped over in the sub process
        """
        if self.sync:
            sim.simxSynchronousTrigger(self.clientID)
        sim.simxPauseCommunication(self.clientID, True)

        # get position data
        _, orientation = sim.simxGetObjectOrientation(self.clientID,
                                                      self.camHandle,
                                                      -1,
                                                      sim.simx_opmode_buffer)
        _, position = sim.simxGetObjectPosition(self.clientID,
                                                self.camHandle,
                                                -1,
                                                sim.simx_opmode_buffer)

        # retrieve images
        self.images = self.getImages()
        # only send or receive once images are not none
        if self.images is None:
            time.sleep(1 / 100)
            return

        # send images to depth pipeline
        self.images["display"] = self.Depth2Color(self.images["depth"], 1)
        self.images["display"] = np.vstack([self.images["RGB"], self.images["display"]])
        message = {"position": position, "orientation": orientation, "images": self.images}
        self.queues["output"].send(message, method="recent")
        speeds = self.queues["input"].retrieve(method="no_wait")

        # check keyboard to set wheel directions
        speed_L, speed_R = self.keyboardInput()
        if (speeds is not None) and (not self.keyboard_controlled):
            speed_L = speeds["left"]
            speed_R = speeds["right"]
        # activate gripper
        #if self.gripper_key.click():
        #    self.setGripper("left", -0.05)
        #    self.setGripper("right", 0.05)
        #else:
        #    self.setGripper("left", 0.05)
        #    self.setGripper("right", -0.05)

        # set wheel speeds
        threads = {"left": Thread(target=self.setSpeed, args=("left", speed_L,)),
                   "right": Thread(target=self.setSpeed, args=("right", speed_R,))}
        self.startThreads(threads)
        self.joinThreads(threads)

        sim.simxPauseCommunication(self.clientID, False)

    def getHandle(self, handle):
        """returns handle object from a object within the VREP scene"""
        err, x = sim.simxGetObjectHandle(self.clientID, handle, sim.simx_opmode_oneshot_wait)
        if err != 0:
            raise AssertionError(f"could not get {handle} handle")
        return x

    def Depth2Color(self, depth, maximum):
        depth = (10 / depth).astype(np.uint8)
        return cv2.applyColorMap(cv2.convertScaleAbs(depth, 1), cv2.COLORMAP_JET)

    def cleanup(self):
        # stop vehicle from moving
        for i in range(5):
            self.setSpeed("left", 0)
            self.setSpeed("right", 0)
        # close all queues
        for _, q in self.queues.items():
            q.close()
        # pause simulation and break connection
        sim.simxPauseSimulation(self.clientID, sim.simx_opmode_oneshot)
        sim.simxFinish(-1)
        cv2.destroyAllWindows()

    def startThreads(self, threads):
        """ starts all threads from a dictionary """
        for _, t in threads.items():
            t.start()

    def joinThreads(self, threads):
        """ joins all threads from a dictionary """
        for _, t in threads.items():
            t.join()

    def keyboardInput(self):
        """ reads keyboard inputs and returns corresponding wheel speeds """
        self.keyboard_controlled = self.keyboard_key.click()
        if not self.keyboard_controlled:
            return 0, 0

        if keyboard.is_pressed("right") and keyboard.is_pressed("UP"):
            return self.speed, +self.lr * self.speed
        elif keyboard.is_pressed("left") and keyboard.is_pressed("UP"):
            return +self.lr * self.speed, self.speed
        elif keyboard.is_pressed("UP"):
            return self.speed, self.speed
        elif keyboard.is_pressed("DOWN"):
            return -self.speed, -self.speed
        elif keyboard.is_pressed("left"):
            return -self.lr * self.speed, +self.lr * self.speed
        elif keyboard.is_pressed("right"):
            return +self.lr * self.speed, -self.lr * self.speed
        else:
            return 0, 0

    def setSpeed(self, side, speed, force=500):
        sim.simxSetJointForce(self.clientID,
                              self.motor[side],
                              force,
                              sim.simx_opmode_streaming)
        sim.simxSetJointTargetVelocity(self.clientID,
                                       self.motor[side],
                                       speed,
                                       sim.simx_opmode_streaming)

    def setGripper(self, side, speed, force=500):
        sim.simxSetJointForce(self.clientID, self.gripperHandle[side], force, sim.simx_opmode_streaming)
        sim.simxSetJointTargetVelocity(self.clientID,
                                       self.gripperHandle[side],
                                       speed,
                                       sim.simx_opmode_streaming)

    def getImages(self):
        _, res, rgb = sim.simxGetVisionSensorImage(self.clientID,
                                                   self.camHandle, 0,
                                                   sim.simx_opmode_buffer)
        _, _, depth = sim.simxGetVisionSensorDepthBuffer(self.clientID,
                                                         self.camHandle,
                                                         sim.simx_opmode_buffer)
        if len(rgb) == 0:
            return None
        depth, rgb = [np.array(i) for i in [depth, rgb]]
        rgb = rgb.astype(np.uint8)
        depth.resize([res[1], res[0]])
        depth = cv2.flip(depth, 0)
        rgb.resize([res[1], res[0], 3])
        rgb = cv2.flip(rgb, 0)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        return {"RGB": rgb, "depth": depth}
