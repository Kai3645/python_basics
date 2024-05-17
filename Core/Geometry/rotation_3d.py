import math

import numpy as np

from Core.Basic import NUM_ERROR
from Core.Geometry.Quaternion import Quat

"""
- memo:
	. type of system
		1. Euler
			rot: (roll, pitch, yaw) radians
			order: (list of axis ids) roll -> pitch -> yaw
				=> rotation order is necessary
		2. Rotation
			mat: 3x3 rotation matrix
				=> most useful for calculation
		3. Quaternion
			quat: normalized Quaternion
				=> main used in interpolation
				=> need to normalize for return
		4. AxisAngle
			vector: (vx, vy, vz) normalized
			angle: radians
				=> Warming: this part of code still unstable ..
		4. SO3
			rvec: (rx, ry, rz) 
				=> main used in opencv/slam ..
				=> Warming: this part of code still unstable ..
	. basic conversion roles
+--------------------------------------------+
|                                            |
|       ┌─────────> Euler o─────────┐        |
|       │                           │        |
|       v                           v        |
|    Rotation <───────────────> Quaternion   |
|       ^                           ^        |
|       │                           │        |
|       └───────o AxisAngle <───────┘        |
|                     ^                      | 
|                     |                      |
|                     v                      | 
|                    SO3                     |
|                                            |
+--------------------------------------------+
- support: "┌", "└", "┐", "┘", "─", "│", "├", "┤", "┬", "┴", "┼",

author@Kai3645 left: 
	For fastest performers, input parameters logic will not be checked.
	(ex. quat was not be normalized)
"""


# --------------------> conversion <--------------------

def Euler_Rotation(rot, *, order: tuple = (0, 1, 2)):
	"""
	default = Rz(yaw) * Ry(pitch) * Rx(roll)
	:param: rot: (roll, pitch, yaw) radians
	:param: order: (list of axis ids) roll -> pitch -> yaw
	:return: mat: 3x3 rotation matrix
	"""

	def AngleAxis(angle, idx: int):
		"""
		:param: angle: radians
		:param: idx: 0->x, 1->y, 2->z
		:return: 3x3 rotation matrix
		"""
		cos = math.cos(angle)
		sin = math.sin(angle)
		row = (idx + 1) % 3
		clm = (idx + 2) % 3
		mat = np.eye(3)
		mat[row, row] = cos
		mat[row, clm] = -sin
		mat[clm, row] = sin
		mat[clm, clm] = cos
		return np.asmatrix(mat)

	return np.asarray(AngleAxis(rot[2], order[2]) * AngleAxis(rot[1], order[1]) * AngleAxis(rot[0], order[0]))


def Euler_Quaternion(rot, *, order: tuple = (0, 1, 2)):
	"""
	:param: rot: (roll, pitch, yaw) radians
	:param: order: (list of axis ids) roll -> pitch -> yaw
	:return: quat: normalized Quaternion
	"""

	def AngleAxis(angle, idx: int):
		"""
		:param: angle: radians
		:param: idx: 0->x, 1->y, 2->z
		:return: Quaternion
		"""
		angle *= 0.5
		v = np.zeros(3)
		v[idx] = math.sin(angle)
		w = math.cos(angle)
		return Quat.init_by_wv(w, v)

	quat = AngleAxis(rot[2], order[2]) * AngleAxis(rot[1], order[1]) * AngleAxis(rot[0], order[0])
	return abs(quat).normalization()


def Rotation_Euler(mat, *, order: tuple = (0, 1, 2)):
	"""
	:param: mat: 3x3 rotation matrix
	:param: order: (list of axis ids) roll -> pitch -> yaw
	:return: rot: (roll, pitch, yaw) radians
	"""

	def norm(x: float, y: float):
		return math.sqrt(x * x + y * y)

	sign = np.asmatrix("0 -1 1; 1 0 -1; -1, 1, 0")[order[2], order[0]]

	return np.asarray([
		math.atan2(-sign * mat[order[2], order[1]], mat[order[2], order[2]]),
		math.atan2(sign * mat[order[2], order[0]], norm(mat[order[2], order[1]],
		                                                mat[order[2], order[2]])),
		math.atan2(-sign * mat[order[1], order[0]], mat[order[0], order[0]]),
	], np.float64)


def Rotation_Quaternion(mat):
	"""
	:param: mat: 3x3 rotation matrix
	:return: quat: normalized Quaternion
	"""
	w = math.sqrt(max(0, 1 + mat[0, 0] + mat[1, 1] + mat[2, 2])) * 0.5
	x = math.sqrt(max(0, 1 + mat[0, 0] - mat[1, 1] - mat[2, 2])) * 0.5
	y = math.sqrt(max(0, 1 - mat[0, 0] + mat[1, 1] - mat[2, 2])) * 0.5
	z = math.sqrt(max(0, 1 - mat[0, 0] - mat[1, 1] + mat[2, 2])) * 0.5
	if mat[2, 1] - mat[1, 2] < 0: x = -x
	if mat[0, 2] - mat[2, 0] < 0: y = -y
	if mat[1, 0] - mat[0, 1] < 0: z = -z
	return abs(Quat(w, x, y, z)).normalization()


def Quaternion_Rotation(quat: Quat):
	"""
	:param: quat: normalized Quaternion
	:return: mat: 3x3 rotation matrix
	"""
	mat = np.eye(3)
	mat[0, 0] -= (quat.y * quat.y + quat.z * quat.z) * 2
	mat[1, 1] -= (quat.x * quat.x + quat.z * quat.z) * 2
	mat[2, 2] -= (quat.x * quat.x + quat.y * quat.y) * 2
	mat[0, 1] = (quat.x * quat.y - quat.w * quat.z) * 2
	mat[1, 0] = (quat.x * quat.y + quat.w * quat.z) * 2
	mat[0, 2] = (quat.x * quat.z + quat.w * quat.y) * 2
	mat[2, 0] = (quat.x * quat.z - quat.w * quat.y) * 2
	mat[1, 2] = (quat.y * quat.z - quat.w * quat.x) * 2
	mat[2, 1] = (quat.y * quat.z + quat.w * quat.x) * 2
	return mat


def Quaternion_AxisAngle(quat: Quat):
	"""
	when w == 1 (no rotation happened), return (0, 0, 1), 0 as default
	:param quat: normalized Quaternion
	:return:
		vector: (vx, vy, vz) normalized
		angle: radians
	"""
	if quat.w == 1: return np.asarray((0., 0., 1.)), 0.
	angle = 2 * math.acos(quat.w)
	den = math.sqrt(1 - quat.w * quat.w)
	vector = np.asarray([quat.x / den, quat.y / den, quat.z / den])
	vector /= np.linalg.norm(vector)
	return vector, angle


def AxisAngle_Rotation(vector, angle: float):
	"""
	:param: vector: (vx, vy, vz) normalized
	:param: angle: radians
	:return: mat: 3x3 rotation matrix
	"""
	c = math.cos(angle)
	s = math.sin(angle)
	# vector /= np.linalg.norm(vector)
	sign = np.asmatrix([
		[0, -vector[2], vector[1]],
		[vector[2], 0, -vector[0]],
		[-vector[1], vector[0], 0]
	], np.float64)
	x = np.asmatrix(vector)
	return np.asarray(c * np.eye(3) + (1 - c) * (x.T * x) + s * sign)


def AxisAngle_Quaternion(vector, angle: float):
	"""
	:param vector: (vx, vy, vz) normalized
	:param angle: radians
	:return: quat: normalized Quaternion
	"""
	a = angle * 0.5
	v = math.sin(a) * np.asarray(vector)
	w = math.cos(a)
	return abs(Quat.init_by_wv(w, v)).normalization()


def SO3_AxisAngle(rvec):
	"""
	:param rvec: (rx, ry, rz)
	:return:
		vector: (vx, vy, vz) normalized
		angle: radians
	"""
	angle = np.linalg.norm(rvec)
	if angle < NUM_ERROR: return np.asarray((0., 0., 1.)), 0.
	vector = np.asarray(rvec) / angle
	return vector, angle


def AxisAngle_SO3(vector, angle: float):
	"""
	:param vector: (vx, vy, vz) normalized
	:param angle: radians
	:return: rvec: (rx, ry, rz)
	"""
	return np.asarray(vector) * angle


# indirect
def Euler_AxisAngle(rot, *, order: tuple = (0, 1, 2)):
	"""
	Euler -> Quaternion -> AxisAngle
	:param rot: (roll, pitch, yaw) radians
	:param order: (list of axis ids) roll -> pitch -> yaw
	:return:
		vector: (vx, vy, vz) normalized
		angle: radians
	"""
	quat = Euler_Quaternion(rot, order = order)
	return Quaternion_AxisAngle(quat)


# indirect
def Rotation_AxisAngle(mat):
	"""
	Rotation -> Quaternion -> AxisAngle
	:param mat: 3x3 rotation matrix
	:return:
		vector: (vx, vy, vz) normalized
		angle: radians
	"""
	quat = Rotation_Quaternion(mat)
	return Quaternion_AxisAngle(quat)


# indirect
def Quaternion_Euler(quat: Quat, *, order: tuple = (0, 1, 2)):
	"""
	Quaternion -> Rotation -> Euler
	:param: quat: normalized Quaternion
	:param: order: (list of axis ids) roll -> pitch -> yaw
	:return: rot: (roll, pitch, yaw) radians
	"""
	mat = Quaternion_Rotation(quat)
	return Rotation_Euler(mat, order = order)


# indirect
def AxisAngle_Euler(vector, angle: float, *, order: tuple = (0, 1, 2)):
	"""
	AxisAngle -> Rotation -> Euler
	:param vector: (vx, vy, vz) normalized
	:param angle: radians
	:param order: (list of axis ids) roll -> pitch -> yaw
	:return: rot: (roll, pitch, yaw) radians
	"""
	mat = AxisAngle_Rotation(vector, angle)
	return Rotation_Euler(mat, order = order)


# indirect
def SO3_Rotation(rvec):
	"""
	SO3 -> AxisAngle -> Rotation
	:param rvec: (rx, ry, rz)
	:return: mat: 3x3 rotation matrix
	"""
	vector, angle = SO3_AxisAngle(rvec)
	return AxisAngle_Rotation(vector, angle)


# indirect
def Rotation_SO3(mat):
	"""
	:param mat: 3x3 rotation matrix
	:return: rvec: (rx, ry, rz)
	"""
	vector, angle = Rotation_AxisAngle(mat)
	return AxisAngle_SO3(vector, angle)


# indirect
def SO3_Euler(rvec, *, order: tuple = (0, 1, 2)):
	"""
	SO3 -> AxisAngle -> Rotation -> Euler
	:param rvec: (rx, ry, rz)
	:param order: (list of axis ids) roll -> pitch -> yaw
	:return: rot: (roll, pitch, yaw) radians
	"""
	vector, angle = SO3_AxisAngle(rvec)
	mat = AxisAngle_Rotation(vector, angle)
	return Rotation_Euler(mat, order = order)


# indirect
def Euler_SO3(rot, *, order: tuple = (0, 1, 2)):
	"""
	Euler -> Quaternion -> AxisAngle -> SO3
	:param rot: (roll, pitch, yaw) radians
	:param order: (list of axis ids) roll -> pitch -> yaw
	:return: rvec: (rx, ry, rz)
	"""
	quat = Euler_Quaternion(rot, order = order)
	vector, angle = Quaternion_AxisAngle(quat)
	return AxisAngle_SO3(vector, angle)


# indirect
def SO3_Quaternion(rvec):
	"""
	SO3 -> AxisAngle -> Quaternion
	:param rvec: (rx, ry, rz)
	:return: quat: normalized Quaternion
	"""
	vector, angle = SO3_AxisAngle(rvec)
	return AxisAngle_Quaternion(vector, angle)


# indirect
def Quaternion_SO3(quat: Quat):
	"""
	Quaternion -> AxisAngle -> SO3
	:param quat: normalized Quaternion
	:return: rvec: (rx, ry, rz)
	"""
	vector, angle = Quaternion_AxisAngle(quat)
	return AxisAngle_SO3(vector, angle)


# --------------------> interpolation <--------------------

def Euler_Interpolate(Rots, idxes1, idxes2, t, *, order: tuple = (0, 1, 2)):
	"""
	:param: Rot: basic Euler array [rad]
	:param: idxes1: left indexes(int) id-array
	:param: idxes2: right indexes(int) id-array
	:param: t: required scaling 1d-array in [0, 1]
	:param: order: roll -> pitch -> yaw
	:return:
	"""
	Qs = np.asarray([Euler_Quaternion(r, order = order) for r in Rots])
	Qs = list(map(Quat.slerp, Qs[idxes1], Qs[idxes2], t))
	return np.asarray([Quaternion_Euler(q, order = order) for q in Qs])

# todo: 1. add analyze funcs
# todo: 2. add test funcs for debug
