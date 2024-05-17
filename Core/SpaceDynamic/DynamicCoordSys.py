import numpy as np

from Core.Geometry import CoordSys


class DCoordSys(CoordSys):
	# Coordinate System(cs) Initial Convert Matrix(icm)
	w2v = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], np.float64)
	v2w = w2v.transpose()
	v2c = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]], np.float64)
	c2v = v2c.transpose()
	w2c = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], np.float64)
	c2w = w2c.transpose()
	E = np.eye(3)

	# part 01
	@classmethod
	def convW2A(cls, rot, w2a = E):
		"""
		rotate order: X(roll) -> Y(pitch) -> Z(yaw)
		Euler angles code "ZYX"
		:param rot: (3, ) array like, any cs Euler rotation in world cs, named (roll, pitch, yaw)[rad]
		:param w2a: (3, 3) array like, cs icm
		:return:
			R: (3, 3) array like, rotation
		"""
		return cls.conv(rot, order = (0, 1, 2)).dot(w2a)

	@classmethod
	def deconvW2A(cls, R, a2w = E):
		"""
		:param R: (3, 3) array like, rotation
		:param a2w: (3, 3) array like, cs icm
		:return:
			rot: (3, ) array like, any cs Euler rotation in world cs, named (roll, pitch, yaw)[rad]
		"""
		return cls.deconv(np.dot(R, a2w), order = (0, 1, 2))

	@classmethod
	def convV2A(cls, rot, v2a = E):
		"""
		rotate order: X(roll) -> Z(pitch) -> Y(yaw)
		Euler angles code "YZX"
		:param rot: (3, ) array like, any cs Euler rotation in vehicle cs, named (roll, pitch, yaw)[rad]
		:param v2a: (3, 3) array like, cs icm
		:return:
			R: (3, 3) array like, rotation
		"""
		return cls.conv(rot, order = (0, 2, 1)).dot(v2a)

	@classmethod
	def deconvV2A(cls, R, a2v = E):
		"""
		:param R: (3, 3) array like, rotation
		:param a2v: (3, 3) array like, cs icm
		:return:
			rot: (3, ) array like, any cs Euler rotation in vehicle cs, named (roll, pitch, yaw)[rad]
		"""
		return cls.deconv(np.dot(R, a2v), order = (0, 2, 1))

	@classmethod
	def convC2A(cls, rot, c2a = E):
		"""
		rotate order: Z(roll) -> X(pitch) -> Y(yaw)
		Euler angles code "ZXY"
		:param rot: (3, ) array like, any cs Euler rotation in camera cs, named (roll, pitch, yaw)[rad]
		:param c2a: (3, 3) array like, cs icm
		:return:
			R: (3, 3) array like, rotation
		"""
		return cls.conv(rot, order = (2, 0, 1)).dot(c2a)

	@classmethod
	def deconvC2A(cls, R, a2c = E):
		"""
		:param R: (3, 3) array like, rotation
		:param a2c: (3, 3) array like, cs icm
		:return:
			rot: (3, ) array like, any cs Euler rotation in camera cs, named (roll, pitch, yaw)[rad]
		"""
		return cls.deconv(np.dot(R, a2c), order = (2, 0, 1))

	# part 02
	@classmethod
	def convW2V(cls, rot):
		return cls.convW2A(rot, cls.w2v)

	@classmethod
	def deconvW2V(cls, R):
		return cls.deconvW2A(R, cls.v2w)

	@classmethod
	def convW2C(cls, rot):
		return cls.convW2A(rot, cls.w2c)

	@classmethod
	def deconvW2C(cls, R):
		return cls.deconvW2A(R, cls.c2w)

	@classmethod
	def convV2W(cls, rot):
		return cls.convV2A(rot, cls.v2w)

	@classmethod
	def deconvV2W(cls, R):
		return cls.deconvV2A(R, cls.w2v)

	@classmethod
	def convV2C(cls, rot):
		return cls.convV2A(rot, cls.v2c)

	@classmethod
	def deconvV2C(cls, R):
		return cls.deconvV2A(R, cls.c2v)

	@classmethod
	def convC2V(cls, rot):
		return cls.convC2A(rot, cls.c2v)

	@classmethod
	def deconvC2V(cls, R):
		return cls.deconvC2A(R, cls.v2c)

	@classmethod
	def convC2W(cls, rot):
		return cls.convC2A(rot, cls.c2w)

	@classmethod
	def deconvC2W(cls, R):
		return cls.deconvC2A(R, cls.w2c)
