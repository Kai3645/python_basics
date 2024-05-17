import math

import numpy as np


def angle_points3(point_0, point_1, point_2):
	vector_1 = np.asarray(point_1, np.float64) - point_0
	vector_2 = np.asarray(point_2, np.float64) - point_0
	vector_1 /= np.linalg.norm(vector_1)
	vector_2 /= np.linalg.norm(vector_2)
	return math.acos(np.dot(vector_1, vector_2))
