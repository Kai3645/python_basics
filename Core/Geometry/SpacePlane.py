import math

import numpy as np

from Core.Basic import num2str, NUM_ERROR, is_zero, errInfo
from Core.Math import gaussian_elimination


class SpacePlane:
	"""
	(x, y, z) = p + t1 * v1 + t2 * v2
	nv * X = d
	dimension = 3
	degree of freedom = 2
	"""

	def __init__(self, nv, d):
		"""
		:param nv: normal vector
		:param d: distance of zero to plane
		"""
		assert d >= 0, ">> err, negative plane distance .."
		self.nv = np.array(nv, np.float64)
		self.d = d

	def __str__(self):
		tmp_str = "plane, "
		tmp_str += f"({num2str(self.nv, 4)}), "
		tmp_str += f"{num2str(self.d, 4)}"
		return tmp_str

	def __eq__(self, other):
		if not is_zero(self.d - other.d): return False
		vv = abs(np.dot(self.nv, other.nv))
		if not is_zero(vv - 1): return False
		return True

	# ---------- initial functions ----------
	@classmethod
	def init_pn(cls, p, nv):
		rr = nv[0] * nv[0] + nv[1] * nv[1] + nv[2] * nv[2]
		assert rr > NUM_ERROR, errInfo(">> err, zero plane normal ..")
		r = math.sqrt(rr)
		rd = p[0] * nv[0] + p[1] * nv[1] + p[2] * nv[2]
		if rd < 0: r = -r
		return cls([nv[0] / r, nv[1] / r, nv[2] / r], rd / r)

	@classmethod
	def init_p3(cls, p1, p2, p3):
		v1 = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]]
		v2 = [p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]]
		return cls.init_pn(p1, np.cross(v1, v2))

	@classmethod
	def init_nd(cls, nv, d):
		rr = nv[0] * nv[0] + nv[1] * nv[1] + nv[2] * nv[2]
		assert rr > NUM_ERROR, errInfo(">> err, zero plane normal ..")
		r = math.sqrt(rr)
		if d < 0:
			r = -r
			d = -d
		return cls([nv[0] / r, nv[1] / r, nv[2] / r], d)

	# ---------- functions ----------
	def project(self, pts):
		"""
		P_proj = P - D * nv
		:param pts: (n, 3) array like
		:return:
		"""
		P = np.atleast_2d(pts)
		D = self.distance(pts)
		K = np.tile(D, (3, 1)).transpose()
		return P - K * self.nv

	def distance(self, pts, force_positive = False):
		"""
		D = P * nv - d
		:param pts:
		:param force_positive:
		:return:
		"""
		P = np.atleast_2d(pts)
		D = np.dot(P, self.nv) - self.d
		if not force_positive: return D
		return np.abs(D)

	def get_pv2(self):
		M = np.asarray([self.nv[0], self.nv[1], self.nv[2], self.d])
		p, dof, W = gaussian_elimination(M, fast_mode = False)
		return p, W[0], W[1]

	def adapt_sample(self, pts = None):
		p0, wx, wy = self.get_pv2()

		total = 7
		scale = 1
		if pts is not None:
			pts = self.project(pts)
			if len(pts) == 1:
				p0 = pts[0]
			else:
				p0 = np.mean(pts, axis = 0)
				pts -= p0
				rr = np.max(np.sum(pts * pts, axis = 1))
				scale = math.sqrt(rr) * 1.2
		K = np.tile(np.linspace(-scale, scale, total), (3, 1)).transpose()
		pt1 = np.tile(p0, (total * 2, 1))
		pt2 = np.copy(pt1)
		pt1[:total] += K * wx + wy * scale
		pt2[:total] += K * wx - wy * scale
		pt1[total:] += K * wy + wx * scale
		pt2[total:] += K * wy - wx * scale
		return pt1, pt2


if __name__ == '__main__':
	from Core.Visualization import KaisColor


	def main():
		pts1 = np.random.random((10, 3))
		sPlane = SpacePlane.init_nd([1, 1, 1], 1)
		pts2 = sPlane.project(pts1)
		Ds = sPlane.distance(pts1)
		print(Ds)

		folder = "/home/kai/PycharmProjects/pyCenter/d_2022_0718/"

		fw = open(folder + "pts.csv", "w")
		color = KaisColor.plotColor("crimson")
		col_str = num2str(color, 3, separator = ",")
		for p in pts1:
			tmp_str = num2str(p, 3, separator = ",")
			fw.write(tmp_str + "," + col_str + "\n")
		color = KaisColor.plotColor("forestgreen")
		col_str = num2str(color, 3, separator = ",")
		for p in pts2:
			tmp_str = num2str(p, 3, separator = ",")
			fw.write(tmp_str + "," + col_str + "\n")
		fw.close()

		fw = open(folder + "pts_line.obj", "w")
		for i, (p1, p2) in enumerate(zip(pts1, pts2)):
			fw.write("v " + num2str(p1, 3, separator = " ") + "\n")
			fw.write("v " + num2str(p2, 3, separator = " ") + "\n")
			fw.write(f"l {i * 2 + 1} {i * 2 + 2}\n")
		fw.close()
		pass


	main()
	pass
