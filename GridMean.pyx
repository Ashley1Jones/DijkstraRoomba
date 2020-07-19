import numpy as np
cimport numpy as np
cimport cython
import heapq
import math


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float_t, ndim=1] C_svd(np.ndarray[np.float_t, ndim=3] groups):

    # define types
    cdef Py_ssize_t i, n_points
    cdef np.ndarray[np.float_t, ndim=2] group
    cdef np.ndarray[np.float_t, ndim=1] normal, dots

    n_points = groups.shape[2]
    dots = np.ones(n_points, dtype=np.float64)

    for i in range(n_points):
        # select the indexed group
        group = groups[:, :, i]
        normal = np.linalg.svd(group)[0][:, 2]
        dots[i] = np.abs(np.dot(normal, [0, 1, 0]))

    return dots

cpdef C_padding(np.ndarray[np.uint8_t, ndim=2] code_map,
        int value1,
        int value2,
        int pad):

    shape = code_map.shape
    cdef Py_ssize_t max_rows = shape[0], max_cols = shape[1], i, j
    cdef int val

    for i in range(pad, max_rows-pad):
        for j in range(pad, max_cols-pad):
            val = code_map[i, j]
            if val == value1:
                code_map[i-pad:i+pad, j-pad:j+pad] = value2

    return code_map

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef C_GridMean(np.ndarray[np.int_t, ndim=2] indices,
    np.ndarray[np.float_t, ndim=1] angles,
    np.ndarray[np.int_t, ndim=2] count_map,
    np.ndarray[np.float_t, ndim=2] angle_map):

    cdef int i, j, imax, jmax
    cdef np.int_t count, index0, index1

    imax = indices.shape[1]

    for i in range(imax):
        index0 = indices[0, i]
        index1 = indices[1, i]
        count = count_map[index0, index1]
        angle_map[index0, index1] = (angle_map[index0, index1]*count+angles[i])/(count+1)
        count_map[index0, index1] += 1

    return count_map, angle_map


cdef float heuristic(int row1, int col1, int row2, int col2):
    return (row1 - row2) ** 2 + (col1 - col2) ** 2


cpdef C_Astar(np.ndarray[np.uint8_t, ndim=2] array, int start_row, int start_col, int goal_row, int goal_col, int obstacle, int pad):
    # http://code.activestate.com/recipes/578919-python-a-pathfinding-with-binary-heap/
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    close_set = set()
    came_from = {}
    gscore = {(start_row, start_col): 0}
    fscore = {(start_row, start_col): heuristic(start_row, start_col, goal_row, goal_col)}
    oheap = []
    heapq.heappush(oheap, (fscore[(start_row, start_col)], (start_row, start_col)))

    cdef int current_row, current_col, i, j, val
    cdef float tentative_g_score, current_gscore
    cdef int max_rows = array.shape[0] - 1
    cdef int max_cols = array.shape[1] - 1

    while oheap:
        current_row, current_col = heapq.heappop(oheap)[1]
        current = (current_row, current_col)
        if (current_row == goal_row) and (current_col == goal_col):
            data = []
            while (current_row, current_col) in came_from:
                data.append([current_row, current_col])
                current_row, current_col = came_from[(current_row, current_col)]
            data = data + [[start_row, start_col]]
            data = data[::-1]
            data = np.array(data).T
            return data, fscore[(goal_row, goal_col)]

        close_set.add((current_row, current_col))
        current_gscore = gscore[current]
        for i, j in neighbors:
            neighbor_row = current_row + i
            neighbor_col = current_col + j
            neighbor = (neighbor_row, neighbor_col)
            tentative_g_score = current_gscore + abs(current_row - neighbor_row) + abs(current_col - neighbor_col)# heuristic(current_row, current_col, neighbor_row, neighbor_col)
            #tentative_g_score = current_gscore + heuristic(goal_row, goal_col, neighbor_row, neighbor_col)
            if 0 < neighbor_row < max_rows:
                if 0 <= neighbor_col < max_cols:
                    val = array[neighbor_row][neighbor_col]
                    if val == obstacle:
                        continue
                    if val == pad:
                        # add extra X cost if element is padded obstacle
                        tentative_g_score += 500
                else:
                    # array bound y walls
                    continue
            else:
                # array bound x walls
                continue
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue

            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [p[1] for p in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor_row, neighbor_col, goal_row, goal_col)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))

    return None


cpdef C_disktra(np.ndarray[np.uint8_t, ndim=2] grid,
            np.ndarray[np.float_t, ndim=2] angle_grid,
            np.ndarray[np.float_t, ndim=2] proximity_cost,
            int unsearched,
            int explored,
            int obstacle,
            int pad):
        """
        uses greedy search to find the closest unexplored space
        """
        # define all datatypes
        cdef Py_ssize_t max_rows, max_cols, start_row, start_col, current_row, current_col, neighbor_row, neighbor_col
        cdef np.ndarray[np.float_t, ndim=2] distances
        cdef int found, val
        cdef float distance, val1

        max_rows = grid.shape[0]
        max_cols = grid.shape[1]
        start_row = int(max_rows / 2)
        start_col = int(max_cols / 2)
        distances = np.zeros([max_rows, max_cols], dtype=np.float64)
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
                    if val == unsearched:
                        found += 1
                    if found > 100:
                        distances[np.where(distances == 0)] = 255
                        distances = distances + angle_grid + proximity_cost
                        distances[np.where(grid == explored)] = 500
                        distances[np.where(grid == unsearched)] -= 5
                        min_pos = np.where(distances == distances.min())
                        current_row, current_col = min_pos[0][0], min_pos[1][0]

                        route = []
                        while (current_row, current_col) in came_from:
                            route.append([current_row, current_col])
                            current_row, current_col = came_from[(current_row, current_col)]
                        route = route + [[start_row, start_col]]
                        route = route[::-1]
                        route = np.array(route).T
                        return distances, route, (min_pos[0][0], min_pos[1][0])

        return distances, None, None





