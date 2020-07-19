import numpy as np
import math
import time
from GridMean import *
import cv2
from Parameters import *

class MappingTemplate(object):
    def __init__(self, size, large_size=40, scale=0.1):
        """
        this parent class is used as a template for all other classes needing a map
        :param size: float, or int of map real world size
        """
        self.size = size
        self.large_size = large_size
        # scale is size of each grid cell
        self.scale = 0.1
        self.small_num_cells = math.ceil(self.size / self.scale)
        self.small_map = np.zeros([self.small_num_cells, self.small_num_cells])
        self.small_count = np.zeros([self.small_num_cells, self.small_num_cells], dtype=np.int)
        self.small_coded = np.zeros([self.small_num_cells, self.small_num_cells], dtype=np.uint8)
        # create large maps
        self.large_num_cells = math.ceil(self.large_size / self.scale)
        self.large_map = np.zeros([self.large_num_cells, self.large_num_cells])
        self.large_count = np.zeros([self.large_num_cells, self.large_num_cells], dtype=np.int)
        self.large_coded = np.zeros([self.large_num_cells, self.large_num_cells], dtype=np.uint8)
        # mid point of matrices of large matrices
        self.mid = math.ceil(self.large_size / (2 * self.scale))
        # half size of small matrices
        self.half_small = math.ceil(self.size / (2 * self.scale))
        # create codes for maps
        self.map_codes = MapCodes

    def snippet(self, pos_row, pos_col, large_mat):
        """
        takes a small portion of large map
        :param pos_row: mid point row of where small mat should be taken
        :param pos_col: same as above
        :param large_mat: matrix which is taken from
        :return: smaller square matrix of shape "half small * 2"
        """
        return large_mat[pos_row - self.half_small:pos_row + self.half_small,
                    pos_col - self.half_small:pos_col + self.half_small]

    def place(self, pos_row, pos_col, small_mat, large_mat):
        """
        Same as snippet but in reverse
        """
        large_mat[pos_row - self.half_small:pos_row + self.half_small,
        pos_col - self.half_small:pos_col + self.half_small] = small_mat
        return large_mat

    def convert_coords(self, x, z, mat, values=None):
        """
        :param x: horizontal coordinate converted into column
        :param z: vertical coordinate converted into row
        :param mat: matrix to which the cartesian coords are applied to
        :param values: if values are linked with coordinates,
            they also need to be linked to filtered coordinates
        :return: ndarray of rows and cols as well as filtered values if included
        """
        max_row, max_col = mat.shape
        row = (max_row / 2) - (z / self.scale)
        col = (max_col / 2) + (x / self.scale)
        if type(row) == np.ndarray:
            # check whether rows and cols are within bounds
            row_within_bounds = np.where(row < (max_row - 1))
            row, col = [arr[row_within_bounds] for arr in [row, col]]
            if values is not None:
                values = values[row_within_bounds]
            col_within_bounds = np.where(col < (max_col - 1))
            row, col = [arr[col_within_bounds] for arr in [row, col]]
            if values is not None:
                values = values[col_within_bounds]
        else:
            return int(row), int(col)

        if values is not None:
            return row.astype(np.int), col.astype(np.int), values
        else:
            return row.astype(np.int), col.astype(np.int)


class AngleMapper(MappingTemplate):
    def __init__(self, point_range):
        self.size = point_range * 2
        MappingTemplate.__init__(self, self.size)
        # create image for debugging
        self.small_coded_vis = None
        # used for navigation
        self.position = []
        # take cython grid mean
        self.C_GridMean = C_GridMean

    def update(self, x, z, angles, orientation, position):
        start = time.time()
        pos_row, pos_col = self.convert_coords(position[0], position[1], self.large_map)
        # take smaller snippets of large maps
        self.small_map = self.snippet(pos_row, pos_col, self.large_map)
        self.small_count = self.snippet(pos_row, pos_col, self.large_count)
        self.small_coded = self.snippet(pos_row, pos_col, self.large_coded)

        # place angles onto the map with cython written loop which averages values sharing same cell
        rows, cols, angles = self.convert_coords(x, z, self.small_map, angles)
        indices = np.array([rows, cols])
        self.small_count, self.small_map = self.C_GridMean(indices, angles, self.small_count, self.small_map)

        # paint the coded map
        self.small_coded[np.where(self.small_count > 0)] = self.map_codes.explored
        self.small_coded[np.where(self.small_map > 0.7)] = self.map_codes.obstacle

        # place snippets back onto map
        self.large_map = self.place(pos_row, pos_col, self.small_map, self.large_map)
        self.large_count = self.place(pos_row, pos_col, self.small_count, self.large_count)
        self.large_coded = self.place(pos_row, pos_col, self.small_coded, self.large_coded)
        self.position = [pos_row, pos_col]
