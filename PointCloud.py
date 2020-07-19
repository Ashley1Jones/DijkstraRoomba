import numpy as np
import cv2
from math import *
import time
from Mapper import *
from Utils import *
from Navigation import *
from GridMean import *


class PointCloud(object):
    def __init__(self, queue_dict):
        self.queues = queue_dict
        # filter properties
        self.toofar = 3
        self.toohigh = 1
        self.n = 500
        #n = 500
        #self.n = math.floor(math.sqrt(n))

        # create row and column structures
        message = self.queues["input"].retrieve(method="recent")
        self.height, self.width = message["images"]["depth"].shape
        self.rows = np.arange(start=0, stop=self.height)[:, np.newaxis]
        self.rows = np.repeat(self.rows, self.width, axis=1)
        self.cols = np.ones(self.height)[:, np.newaxis] *  np.arange(start=0, stop=self.width)
        #self.rows = np.arange(start=0, stop=self.height)
        #self.rows = np.repeat(self.rows, self.width)
        #self.cols = np.arange(start=0, stop=self.width)
        #self.cols = np.tile(self.cols, self.height)
        #self.cols = self.cols.flatten()

        # init camera
        self.near_clip = 0.1
        self.far_clip = 10
        self.cx = math.floor(self.width / 2)
        self.cy = math.floor(self.height / 2)
        FOV = 90
        self.f = 320 / math.tan(math.pi * (FOV / 2) / 180)

        # grouping
        self.k = 10

        # create mapping class instances
        self.map = AngleMapper(self.toofar)
        self.nav = Navigation(point_range=self.toofar, size=3 * self.toofar)

        # create windows
        self.images_display = NamedWindow("images", size=(500, 500))
        self.angles_display = NamedWindow("angles", place=(500, 0),
                                          size=(500, 500),
                                          conversions="heat_map")
        self.coded_display = NamedWindow("coded", place=(1000, 500), size=(500, 500))
        self.detection_display = NamedWindow("Litter Detection", place=(1000, 0), size=(500, 500))
        # self.large_display = NamedWindow("cost", place=(500, 500), size=(500, 500))

        # find original position of camera
        self.original_orientation = np.array([-1.56821251, 0, 3.14152694])# np.array(message["orientation"])
        self.original_position = np.array(message["position"])

        self.C_svd = C_svd

    def litter_localiser(self, corner, z, scale=0.2):
        # bottom right is index 0 and corners are in cartesian coords
        # not row / col
        # first find center
        br = corner[0]
        tr = corner[1]
        half_box_size = [abs(br[0] - tr[0])/2, abs(br[1] - tr[1])/2]
        half_box_size[0], half_box_size[1] = math.floor(half_box_size[0]*scale), math.floor(half_box_size[1]*scale)
        center = [math.floor((br[0] + tr[0])/2), math.floor((br[1] + tr[1])/2)]
        br = center[0] - half_box_size[0], center[1] - half_box_size[1]
        tr = center[0] + half_box_size[0], center[1] + half_box_size[1]
        z_selected = z[br[1]:tr[1], br[0]:tr[0]]
        rows, cols = self.rows[br[1]:tr[1], br[0]:tr[0]], self.cols[br[1]:tr[1], br[0]:tr[0]]
        # find x and y coords
        x = (cols - self.cx) * z_selected / self.f
        y = (self.cy - rows) * z_selected / self.f
        x, y, z_selected = [np.mean(arr) for arr in [x, y, z_selected]]
        return [x, y, z_selected]

    def main_loop(self):
        # get data from other process
        message = self.queues["input"].retrieve(method="recent")

        start = time.time()
        # convert current depth images to meters
        depth = message["images"]["depth"] * (self.far_clip - self.near_clip) + self.near_clip

        # litter_location = self.litter_localiser(z, message["detections"])

        # reduce size to save computation time
        # create random row and col indices to select random values from 2d arrays
        random_indices = np.array([np.random.randint(low=0, high=self.height, size=self.n),
                                      np.random.randint(low=0, high=self.width, size=self.n)])
        z, rows, cols = self.select([depth, self.rows, self.cols], random_indices, two_dim=True)

        # zero orientation and position from original frame
        euler = np.array(message["orientation"]) - self.original_orientation
        position = np.array(message["position"]) - self.original_position
        R = self.euler2rotation(euler)

        # find position of any detections
        if message["detections"] is not None:
            litter_locations = list(map(lambda corner: self.litter_localiser(corner, depth), message["detections"]))
            litter_locations = [np.matmul(R, np.array(loc)[:, np.newaxis]) for loc in litter_locations]
        else:
            litter_locations = None


        # filter objects too far away
        toofar = np.where(z < self.toofar)
        z, rows, cols = self.select([z, rows, cols], toofar)
        z = z[np.where(z < self.toofar)]

        # find x and y coords
        x = (cols - self.cx) * z / self.f
        y = (self.cy - rows) * z / self.f

        x, y, z = np.matmul(R, np.array([x, y, z]))

        toohigh = np.where(np.abs(y) < self.toohigh)
        x, y, z, rows, cols = self.select([x, y, z, rows, cols], toohigh)

        if z.size > 20:
            time1 = time.time()
            xb = x[:, np.newaxis] - x
            yb = y[:, np.newaxis] - y
            zb = z[:, np.newaxis] - z
            d = (xb ** 2) + (yb ** 2) + (zb ** 2)
            #PrintFunctionTime("\nbroad", time1)

            #time2 = time.time()
            closest = np.argpartition(d, self.k, axis=0)[:self.k]
            #PrintFunctionTime("sorting", time2)

            #p_time = time.time()
            #angles = np.apply_along_axis(self.SVD, 0, closest, x, y, z)
            #PrintFunctionTime("p_time", p_time)

            #c_time = time.time()
            groups = np.array([x[closest], y[closest], z[closest]])
            groups_mean = np.mean(groups, axis=1)
            groups = groups - groups_mean[:, np.newaxis, :]
            #print(groups[:, :, 0])
            #print(groups.shape)
            angles = self.C_svd(groups)
            # dum = np.abs(angles)
            #print(dum)

            angles = np.abs(angles-1)
            #PrintFunctionTime("c_time", time1)


            #print(f"Cloud takes: {round((time.time() - start) * 1000, 3)}ms, FPS: {round(1 / (time.time() - start), 3)}")
            # timing the updates
            start_mn = time.time()
            self.map.update(x, z, angles, euler, position)
            self.nav.update(self.map.large_coded, self.map.position, euler, litter_locations)

            # print(f"Map&Nav takes: {round((time.time() - start_mn) * 1000, 3)}ms, FPS: {round(1 / (time.time() - start_mn), 3)}")

        self.angles_display.show(self.map.small_map, max_value=1)
        self.coded_display.show(self.nav.map_vis)
        self.images_display.show(message["images"]["display"])
        self.detection_display.show(message["images"]["RGB"])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return "break"
        # self.large_display.show(self.nav.distances)

        self.queues["output"].send(self.nav.speed, method="no_wait")


    def euler2rotation(self, euler):
        """
        taken from https://en.wikipedia.org/wiki/Rotation_matrix
        """
        a, b, g = euler[0], euler[1], euler[2]
        return np.array([[cos(a)*cos(b), cos(a)*sin(b)*sin(g)-sin(a)*cos(g), cos(a)*sin(b)*cos(g)+sin(a)*sin(g)],
                        [sin(a)*cos(b), sin(a)*sin(b)*sin(g)+cos(a)*cos(g), sin(a)*sin(b)*cos(g)-cos(a)*sin(g)],
                        [-sin(b), cos(b)*sin(g), cos(b)*cos(g)]])

    def SVD(self, closest, x, y, z):
        group = np.array([x[closest], y[closest], z[closest]])
        centroid = np.mean(group, axis=1)[:, np.newaxis]
        relative = group - centroid
        svd = np.transpose(np.linalg.svd(relative))
        normal = np.transpose(svd[0])[2]
        return np.abs(np.dot(normal, [0, 1, 0]))

    def select(self, arrays, indices, two_dim=False):
        if two_dim:
            return [array[indices[0], indices[1]] for array in arrays]
        else:
            return [array[indices] for array in arrays]

    def cleanup(self):
        # close all queues
        for _, q in self.queues.items():
            q.close()


