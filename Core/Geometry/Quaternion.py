import math

import numpy as np

"""
- Reference:
	1. T.Akenine-Möller, "Real-Time Rendering, 4th", 2018
	2. "www.thetenthplanet.de/archives/1994"
	3. "https://www.geometrictools.com/Documentation/FastAndAccurateSlerp.pdf"
"""

# Constants
SLERP_threshold = 0.9995
SLERP_mu = 1.85298109240830  # Ref. 3 Table 2
SLERP_U = np.asarray([  # 1/((2i+1)i)
	1 / 3, 1 / 10, 1 / 21, 1 / 36, 1 / 55, 1 / 78, 1 / 105, SLERP_mu / 136
], np.float64)
SLERP_V = np.asarray([  # i/(2i+1)
	1 / 3, 2 / 5, 3 / 7, 4 / 9, 5 / 11, 6 / 13, 7 / 15, SLERP_mu * 8 / 17
], np.float64)


class Quaternion:
	def __init__(self, w, x, y, z):
		self.w = float(w)
		self.x = float(x)
		self.y = float(y)
		self.z = float(z)

	@classmethod
	def init_by_wv(cls, w, v):
		return cls(w, v[0], v[1], v[2])

	@classmethod
	def init_by_array(cls, a):
		return cls(a[0], a[1], a[2], a[3])

	@classmethod
	def scalar(cls, w):
		return cls(w, 0, 0, 0)

	def __repr__(self):
		return f'Quat({self.w:.6f}, {self.x:.6f}, {self.y:.6f}, {self.z:.6f})'

	def __copy__(self):
		return self.__class__(self.w, self.x, self.y, self.z)

	def __pos__(self):
		return self.__copy__()

	def __neg__(self):
		return self.__class__(-self.w, -self.x, -self.y, -self.z)

	def __abs__(self):
		# let w >= 0
		if self.w < 0: return -self
		return self

	def __add__(self, other):
		return self.__class__(
			self.w + other.w,
			self.x + other.x,
			self.y + other.y,
			self.z + other.z
		)

	def __sub__(self, other):
		return self.__class__(
			self.w - other.w,
			self.x - other.x,
			self.y - other.y,
			self.z - other.z
		)

	def __mul__(self, other):
		w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
		x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
		y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
		z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
		return self.__class__(w, x, y, z)

	def multiply(self, w: float):  # scalar only
		return self.__class__(self.w * w, self.x * w, self.y * w, self.z * w)

	def divide(self, w: float):  # scalar only
		return self.multiply(1 / w)

	def dot(self, other):
		return self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z

	def norm(self):
		return math.sqrt(self.dot(self))

	def conjugate(self):
		return self.__class__(self.w, -self.x, -self.y, -self.z)

	def normalization(self):
		return self.divide(self.norm())

	def inv(self):
		return self.conjugate().divide(self.dot(self))

	def exp(self):
		# todo:
		pass

	def log(self):
		# todo:
		pass

	def asarray(self, dtype = np.float64):
		return np.asarray([self.w, self.x, self.y, self.z], dtype)

	@classmethod
	def slerp_(cls, quat1, quat2, t):
		cos = quat1.dot(quat2)
		if cos < 0:
			quat1 = -quat1
			cos = -cos

		if cos > SLERP_threshold:
			quat = quat1 + (quat2 - quat1).multiply(t)
			return abs(quat).normalization()

		theta = math.acos(cos)
		theta_t = theta * t
		sin = math.sin(theta)
		sin_t = math.sin(theta_t)
		cos_t = math.cos(theta_t)

		f2 = sin_t / sin
		f1 = cos_t - cos * f2
		quat = quat1.multiply(f1) + quat2.multiply(f2)
		return abs(quat).normalization()

	@classmethod
	def slerp(cls, quat1, quat2, t):
		"""
		Fast And Accurate Slerp ?
		:param: quat1:
		:param: quat2:
		:param: t:
		:return:
		"""
		cos = quat1.dot(quat2)
		if cos < 0:
			quat1 = -quat1
			cos = -cos

		xm1 = cos - 1
		d = 1 - t
		sqr_t = t * t
		sqr_d = d * d
		t_coe = (SLERP_U * sqr_t - SLERP_V) * xm1
		d_coe = (SLERP_U * sqr_d - SLERP_V) * xm1
		for i in range(1, 8):
			t_coe[i] *= t_coe[i - 1]
			d_coe[i] *= d_coe[i - 1]
		f2 = (1 + np.sum(t_coe)) * t
		f1 = (1 + np.sum(d_coe)) * d
		quat = quat1.multiply(f1) + quat2.multiply(f2)
		return abs(quat).normalization()

	@classmethod
	def squad(cls):
		# todo: cubic spline
		pass

	pass


Quat = Quaternion

if __name__ == '__main__':
	def test():
		data = (np.random.random((10, 4)) - 0.5) * 10
		for row in data:
			x = Quat.init_by_array(row)
			print(x)

		pass


	test()
