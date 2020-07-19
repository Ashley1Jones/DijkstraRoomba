import numpy as np
import cv2
from Mapper import *
import math
import heapq
from Utils import *


class Navigation(MappingTemplate):
    def __init__(self, point_range, size):
        """
        create navigation as a child class given it needs to ...
        inherit some properties from mapper
        """
        MappingTemplate.__init__(self, size)
        # init parameters
        self.cost_codes = CostCodes()

        self.range = point_range
        # pad obstacles by x amount and convert to grid cells
        self.obstacle_padding = math.ceil(0.4 / self.scale)
        # windows need to be odd therefore, make odd if even
        if self.obstacle_padding % 2 == 0:
            self.obstacle_padding += 1
        self.half_pad = math.floor(self.obstacle_padding / 2)

        # init wheel speeds
        self.speed = {'left': 0, 'right': 0}
        # init values that need to be carried from last loop
        self.last_pos = None
        self.last_route = None
        self.last_beta = None

        # init cost map values
        self.distance_cost = self.create_distance_cost(self.half_small)
        self.angle_cost = self.create_angle_cost(self.half_small)
        self.proximity_cost = self.create_prox_cost(self.half_small)

        # init parameters needed for proximity
        self.prox_radius = math.floor(0.4 * self.range / self.scale)

        self.C_Astar = C_Astar
        self.C_disktra = C_disktra
        self.C_padding = C_padding

    def get_distances(self, grid, obstacle, pad, angle_grid):
        """
        uses greedy search to find the closest unexplored space
        """
        start = time.time()
        max_rows, max_cols = grid.shape
        start_row = int(max_rows / 2)
        start_col = int(max_cols / 2)
        distances = np.zeros([max_rows, max_cols])
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        current_row, current_col = start_row, start_col
        distances[current_row, current_col] = 1
        current_distance = 1
        oheap = []
        heapq.heappush(oheap, (current_distance, (start_row, start_col)))
        came_from = {}
        found = 0

        while oheap:
            current_distance, (current_row, current_col) = heapq.heappop(oheap)
            for i, j in neighbors:
                neighbor_row = current_row + i
                neighbor_col = current_col + j
                # check if index is within bounds
                if 0 < neighbor_row < max_rows and 0 <= neighbor_col < max_cols:
                    # if grid cell is occupied then skip
                    val = grid[neighbor_row, neighbor_col]
                    if val == obstacle or val == pad:
                        continue
                    # if grid cell has already been searched, skip
                    val1 = distances[neighbor_row, neighbor_col]
                    if val1 != 0:
                        continue
                    distance = current_distance + abs(current_row - neighbor_row) + abs(current_col - neighbor_col)
                    distances[neighbor_row, neighbor_col] = distance
                    came_from[(neighbor_row, neighbor_col)] = (current_row, current_col)
                    heapq.heappush(oheap, (distance, (neighbor_row, neighbor_col)))
                    # if an unsearched cell is found, return route and find route
                    if val == self.map_codes.unsearched:
                        found += 1
                    if found > 100:
                        #not_zero = np.where(distances != 0)
                        #distances[not_zero] += angle_grid[not_zero]
                        #distances[np.where(distances == 0)] = 255

                        #route = []
                        #while (current_row, current_col) in came_from:
                        #    route.append([current_row, current_col])
                        #    current_row, current_col = came_from[(current_row, current_col)]
                        #route = route + [[start_row, start_col]]
                        #route = route[::-1]
                        #route = np.array(route).T
                        print(f"greedy takes: {round((time.time() - start) * 1000, 3)}ms, FPS: {round(1 / (time.time() - start), 3)}")
                        return distances#, route

        print(f"greedy takes: {round((time.time() - start) * 1000, 3)}ms, FPS: {round(1 / (time.time() - start), 3)}")
        return distances

    def update_cost(self, beta, coded_map):
        """
        creates a cost map where the minimum value is chosen as the destination
        this is done by using dijkstra's algorithm to search for the closest un-searched cell
        """
        # subtract the robots angle around the y axis
        angle_cost = self.angle_cost - beta
        # some sections become higher than 180, therefore adjust them
        angle_cost[np.where(angle_cost > 180)] -= 360
        angle_cost[np.where(angle_cost < -180)] += 360
        angle_cost = np.abs(angle_cost)
        # normalise to value set in parameters
        angle_cost = (angle_cost / angle_cost.max()) * self.cost_codes.angle
        # Proximity is to force the destination a certain amount from the center
        cost = angle_cost + self.distance_cost + self.proximity_cost

        distances, route, min_pos = self.C_disktra(coded_map,
                                   angle_cost,
                                   self.proximity_cost,
                                   self.map_codes.unsearched,
                                   self.map_codes.explored,
                                   self.map_codes.obstacle,
                                   self.map_codes.obstacle_pad)

        self.distances = ((distances / distances.max()) * 255).astype(np.uint8)
        return cost, route, min_pos

    def euler2beta(self, euler):
        """
        used to make up for Vrep's weird euler angles
        """
        alpha = euler[0]
        beta = euler[1]
        if alpha > 3 and beta < 0:
            beta = -math.pi - beta
        elif alpha > 3 and beta > 0:
            beta = math.pi - beta

        return (beta / math.pi) * 180

    def rotation_matrix(self, beta):
        beta = -beta
        beta = (beta / 180) * math.pi
        return np.array([[math.cos(beta), -math.sin(beta)],
                        [math.sin(beta), math.cos(beta)]])

    def draw_line(self, x1, y1, x2, y2, origin=True):
        """
        returns coordinates of line from 1 to 2
        if origin is set to True, x1 is set as origin and coordinates will always originate from 1
        """
        if x1-x2 == 0:
            # if x1 and x2 are the same, infinite gradient
            x = np.ones(abs(y1 - y2)) * x1
            if y1 < y2:
                y = np.arange(y1, y2)
            else:
                y = np.arange(y2, y1)
                y = np.flip(y)
            return np.array([x, y], dtype=np.int)

        if y1-y2 == 0:
            # is y1 and y2 are the same, 0 gradient
            y = np.ones(abs(x1 - x2)) * y1
            if x1 < x2:
                x = np.arange(x1, x2)
            else:
                x = np.arange(x2, x1)
                x = np.flip(x)
            return np.array([x, y], dtype=np.int)

        if abs(x1 - x2) > abs(y1 - y2) or abs(x1 - x2) == abs(y1 - y2):
            # difference in x is greater, or the line is at 45 degrees, use dy/dx as gradient
            grad = (y1 - y2) / (x1 - x2)
            if x1 < x2:
                x = np.arange(x1, x2)
            else:
                x = np.arange(x2, x1)
                x = np.flip(x)
            y = (grad * (x - x1)) + y1
            return np.array([x, y], dtype=np.int)

        if abs(x1 - x2) < abs(y1 - y2):
            # if difference of y is greater, use dx/dy
            # this is used because for the same x cell, there will be multiple y coords
            # this flip solves this
            grad = (x1 - x2) / (y1 - y2)
            if y1 < y2:
                y = np.arange(y1, y2)
            else:
                y = np.arange(y2, y1)
                y = np.flip(y)
            x = (grad * (y - y1)) + x1
            return np.array([x, y], dtype=np.int)

    def create_prox_cost(self, size, weight=200):
        prox = np.zeros([size * 2, size * 2])
        radius = math.floor(0.4 * self.range / self.scale)
        prox = cv2.circle(prox, (size, size),
                          radius=radius, color=weight, thickness=-1)
        return prox

    def create_x_grid(self, size):
        # create ranges and then broadcast to create 2d array of x coords
        xr = np.arange(start=1, stop=size + 1, step=1)
        xl = -np.flip(xr)
        x = np.append(xl, xr)
        x = np.ones((2 * size, 1)) * x
        return x

    def create_y_grid(self, size):
        yb = -np.arange(start=1, stop=size + 1, step=1)
        yt = -np.flip(yb)
        y = np.append(yt, yb)
        y = y[:, np.newaxis] * np.ones(size * 2)
        return y

    def create_distance_cost(self, size):
        """
        :param size: half the size of return array
        :return: returns array of distance costs
        """
        x = np.abs(self.create_x_grid(size))
        # same for y but the broadcasting is in the opposite order
        y = np.abs(self.create_y_grid(size))
        # standard distance calculation
        distance = np.sqrt((x ** 2) + (y ** 2))
        distance = (distance / distance.max()) * self.cost_codes.distance
        return distance.astype(np.uint8)

    def create_angle_cost(self, size, weight=1):
        x = self.create_x_grid(size)
        y = self.create_y_grid(size)
        angles = np.arctan2(y, x)
        # angles = np.abs(angles)
        angles = (angles / math.pi) * 180
        angles = np.rot90(angles)
        angles = np.flip(angles, axis=1)
        return angles

    def update(self, code_map, map_pos, euler, litter_locations):
        """
        overrides mapper update method since we do not need it
        """
        # take snippet of large map
        code_map = self.snippet(map_pos[0], map_pos[1], code_map).copy()

        # before padding, save obstacle locations
        obstacle_locations = np.where(code_map == self.map_codes.obstacle)
        # creat circle around the robot to indicate proximity
        code_map = cv2.circle(code_map, (self.half_small, self.half_small),
                          radius=self.prox_radius, color=self.map_codes.too_close, thickness=-1)
        code_map[obstacle_locations] = self.map_codes.obstacle
        # now use the windowed array to find cells near obstacle
        code_map = self.C_padding(code_map, self.map_codes.obstacle, self.map_codes.obstacle_pad, self.obstacle_padding)

        # place back the saved obstacle locations
        code_map[obstacle_locations] = self.map_codes.obstacle
        # convert vrep's shitty euler angles
        beta = self.euler2beta(euler)
        if self.last_beta is not None:
            if beta - self.last_beta > 45:
                print("sharp change")
                # beta = self.last_beta

        cost, route, min_pos = self.update_cost(beta, code_map)
        ####################################################################
        # Calculate motor speed and directions #
        ####################################################################
        if route is not None:
            # from route, decide which direction to go
            if route.shape[1] > 5:
                direction = np.array([(code_map.shape[0] / 2) - route[0, 5], route[1, 5] - (code_map.shape[1] / 2)])
                direction_angle = np.arctan2(direction[1], direction[0]) * (180 / math.pi)
                # direction it wants to go is the difference between the route and robot
                if direction_angle < 0:
                    direction_angle += 360
                if beta < 0:
                    beta += 360
                a = np.array([direction_angle - beta, direction_angle - beta + 360])
                a = a[np.argmin(np.abs(a))]
                left = (180 + a) / 180
                right = (180 - a) / 180
                self.speed = {'left': left, 'right': right}
        else:
            print("Could not find route")

        ####################################################################
        # create visualisations #
        ####################################################################
        self.cost_vis = (255 * cost / cost.max()).astype(np.uint8)
        # https://stackoverflow.com/questions/51875114/triangle-filling-in-opencv
        # ^^ going to be used for drawing triangles

        self.map_vis = cv2.cvtColor(code_map, cv2.COLOR_GRAY2RGB)
        if route is not None:
            self.map_vis[route[0], route[1], :] = [0, 255, 0]
            if self.last_route is not None:
                self.map_vis[self.last_route[0], self.last_route[1], :] = [0, 255, 255]
            if route.shape[1] > 5:
                self.map_vis[route[0, 5], route[1, 5]] = [0, 0, 255]
            cross = np.array([[0, 0, 2, 1, 0, -1, -2, 0, 0],
                              [-2, -1, 0, 0, 0, 0, 0, 1, 2]], dtype=np.int)
            self.map_vis[min_pos[0]-cross[0], min_pos[1]-cross[1]] = [0, 0, 255]
            if litter_locations is not None:
                for loc in litter_locations:
                    loc = self.convert_coords(loc[0], loc[2], code_map)
                    self.map_vis[loc[0] - cross[0], loc[1] - cross[1]] = [255, 255, 0]
            # self.map_vis[route[0], route[1]] = 255

        self.last_route = route

    def convert_coords_mat2mat(self, pos, mat1, mat2):
        """
        convert coords (row, col) from mat1 to mat2 which of the same scale
        """
        # first find distance from center then add that to new center
        center1 = np.array([mat1.shape[0], mat1.shape[1]])
        center2 = np.array([mat2.shape[0], mat2.shape[1]])
        if pos.size <= 2:
            # if position is a single coordinate
            pos = pos - (center1 / 2)
            pos = pos + (center2 / 2)
        elif pos.size > 2:
            # if there are multiple coordinates in array
            pos = pos - (center1 / 2)[:, np.newaxis]
            pos = pos - (center2 / 2)[:, np.newaxis]
        else:
            # incorrect input
            return None
        return pos.astype(np.int)

    def windowed(self, mat, window_size):
        """
        method used to window 2D-matrix used for convolution
        :param mat: matrix which is split into windows
        :param window_size: size of the window, must be odd
        :return: returns array off shape (window size, nchan, nrow, ncol)
        """
        # get the strides of the original matrix
        row_s, col_s = mat.strides
        # make new strides that are identical, however, are for new shape
        window_strides = (row_s, col_s, row_s, col_s)
        new_shape = (mat.shape[0] - window_size + 1,
                     mat.shape[1] - window_size + 1,
                     window_size,
                     window_size)
        # new strides will create a new array: note that this is not a copy
        windowed = np.lib.stride_tricks.as_strided(mat, shape=new_shape, strides=window_strides)
        # add padding to array with edge values
        windowed = np.pad(windowed,
                          pad_width=((self.half_pad, self.half_pad),
                                     (self.half_pad, self.half_pad),
                                     (0, 0),
                                     (0, 0)),
                          mode="edge")
        return windowed
