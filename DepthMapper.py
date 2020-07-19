import numpy as np
import cv2
from multiprocessing import Queue, Process


def convert_heat_map(array, maximum, background=False):
    if background:
        backdrop = np.where(array == 0)
        array = 255 * (array / maximum)
        array = cv2.applyColorMap(cv2.convertScaleAbs(array, 1), cv2.COLORMAP_JET)
        array[backdrop[0], backdrop[1], :] = [255, 255, 255]
        return array
    else:
        array = 255 * (array / maximum)
        return cv2.applyColorMap(cv2.convertScaleAbs(array, 1), cv2.COLORMAP_JET)


class DepthMapper(Process):
    def __init__(self, finish, Input):
        Process.__init__(self)
        self.input = Input
        self.output = Queue()
        # if this is called break waiting
        self.finish = finish
        # set parameters and find intrinsic stereo parameters
        window_size = 5
        self.min_disp = 1
        self.num_disp = 16 * 7
        self.max_disp = self.min_disp + self.num_disp
        # create stereo object
        self.method = "SGBM"
        if self.method == "SGBM":
            self.stereo = cv2.StereoSGBM_create(minDisparity=self.min_disp,
                                                numDisparities=self.num_disp,
                                                blockSize=16,
                                                P1=8 * 3 * window_size ** 2,
                                                P2=32 * 3 * window_size ** 2,
                                                disp12MaxDiff=1,
                                                uniquenessRatio=10,
                                                speckleWindowSize=100,
                                                speckleRange=32)
        elif self.method == "SAD":
            self.stereo = cv2.StereoBM_create(self.num_disp, 15)

    def getImages(self):
        image = self.input.get()
        for key, img in image.items():
            if key == "depth":
                continue
            image[key] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return image

    def run(self):
        while not self.finish.is_set():
            image = self.getImages()
            # cv2.imshow("both", np.vstack([image["left"], image["right"]]))
            if self.method == "SGBM":
                disparity = self.stereo.compute(image["left"], image["right"]).astype(np.float32) / 16.0
            elif self.method == "SAD":
                disparity = self.stereo.compute(image["left"], image["right"]).astype(np.float32)
            disparity = disparity[:, self.max_disp:]
            image["left"] = image["left"][:, self.max_disp:].astype(np.uint8)
            cv2.imshow("disparity", np.vstack([cv2.cvtColor(image["left"], cv2.COLOR_GRAY2RGB),
                                               convert_heat_map(disparity, self.max_disp)]))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.finish.set()
                break
        self.output.close()
        cv2.destroyAllWindows()
