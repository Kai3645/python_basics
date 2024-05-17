import numpy as np

from Core.Basic import KaisLog
from Core.Geometry.rotation_3d import Euler_Rotation, Rotation_Euler

log = KaisLog.get_log()


class CoordSys:
	@staticmethod
	def mat4d(R, T):
		"""
		mat = | R  tT  |
		      | 0   1  |
		:param R: (3, 3) array like, rotation
		:param T: (3, ) array like, transpose
		:return:
			mat: (4, 4) array like
		"""
		mat = np.eye(4)
		mat[:3, :3] = R
		mat[:3, 3] = T
		return mat

	@staticmethod
	def inv(R, T):
		"""
		R * X + tT = Y
		tR * Y - tR * tT = X
		inv_R = tR
		inv_T = -T * R
		:param R: (3, 3) array like, rotation
		:param T: (3, ) array like, transpose
		:return:
			inv_R: (3, 3) array like, rotation
			inv_T: (3, ) array like, transpose
		"""
		inv_R = np.transpose(R)
		inv_T = -np.dot(T, R)
		return inv_R, inv_T

	@staticmethod
	def trans(R, T, P):
		"""
		tP_ = R * tP + tT
		P_ = P * tR + T
		:param R: (3, 3) array like, rotation
		:param T: (3, ) array like, transpose
		:param P: (n, 3) array like, points
		:return:
			P_: (n, 3) array like, points, transformed
		"""
		P = np.atleast_2d(P)
		P_ = P.dot(np.transpose(R)) + T
		if len(P) > 1: return P_
		return P_[0]

	@staticmethod
	def dot(R1, T1, *args):
		"""
		mat = | R1  tT1 | * | R2  tT2 |
		      | 0    1  |   | 0    1  |
		R = R1 * R2
		tT = R1 * tT2 + tT1
		T = T2 * tR1 + T1
		:param R1: (3, 3) array like, rotation
		:param T1: (3, ) array like, transpose
		:param args: (R2, T2, (Ri, Ti, .. ))
		:return:
			R: (3, 3) array like, rotation
			T: (3, ) array like, transpose
		"""
		n = len(args)
		assert n >= 2 and n % 2 == 0, log.error(f"require (R2, T2, (Ri, Ti, .. )), get len(args) = {n} ..")
		R = R1
		T = T1
		for i in range(0, n, 2):
			Ri = args[i]
			Ti = args[i + 1]
			R = np.dot(R, Ri)
			T = np.dot(Ti, Ri.T) + T
		return R, T

	@staticmethod
	def conv(rot, *, order: tuple = (0, 1, 2)):
		"""
		convert / deconvert
		:param rot: (3, ) array like, new cs Euler rotation in old cs, named (roll, pitch, yaw)[rad]
		:param order: (list of axis ids) roll -> pitch -> yaw
		:return:
			R: (3, 3) array like, rotation
			T: (3, ) array like, transpose
		"""
		return Euler_Rotation(rot, order = order)

	@staticmethod
	def deconv(R, *, order: tuple = (0, 1, 2)):
		"""
		deconvert / convert
		:param R: (3, 3) array like, rotation
		:param order: (list of axis ids) roll -> pitch -> yaw
		:return:
			rot: (3, ) array like, new cs Euler rotation in old cs, named (roll, pitch, yaw)[rad]
		"""
		return Rotation_Euler(R, order = order)
