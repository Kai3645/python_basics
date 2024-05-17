import math

import numpy as np

from Core.Basic import NUM_ERROR, num2str, is_zero, errInfo


class SpaceLine:
	"""
	(x, y, z) = p + t * v
	dimension = 3
	degree of freedom = 1
	for easily using, p and v are mutually perpendicular
	"""

	def __init__(self, p, v):
		self.p = np.array(p, np.float64)
		self.v = np.array(v, np.float64)

	def normalize(self):
		rr = self.v[0] * self.v[0] + self.v[1] * self.v[1] + self.v[2] * self.v[2]
		assert rr > NUM_ERROR, errInfo(">> err, zero line vector ..")
		self.v /= math.sqrt(rr)
		k = self.p[0] * self.v[0] + self.p[1] * self.v[1] + self.p[2] * self.v[2]
		self.p -= k * self.v
		return self

	def __str__(self):
		tmp_str = "line, "
		tmp_str += f"({num2str(self.p, 4)}), "
		tmp_str += f"({num2str(self.v, 4)})"
		return tmp_str

	def __eq__(self, other):
		vv = abs(np.dot(self.v, other.v))
		if not is_zero(vv - 1): return False
		for a in self.p - other.p:
			if not is_zero(a): return False
		return True

	# ---------- initial functions ----------
	@classmethod
	def init_p2(cls, p1, p2):
		v = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
		return cls(p1, v).normalize()

	@classmethod
	def init_pv(cls, p, v):
		return cls(p, v).normalize()

	# ---------- functions ----------
	def project(self, pts):
		"""
		P_proj = p + ((Px - p) * v) * v
		:param pts: (n, 3) array like
		:return:
		"""
		P = np.atleast_2d(pts)
		P = P - self.p
		K = np.tile(np.dot(P, self.v), (3, 1)).transpose()
		return self.p + K * self.v

	def distance(self, pts):
		"""
		:param pts: (n, 3) array like
		:return:
		"""
		P = np.atleast_2d(pts)
		P_proj = self.project(P)
		return np.linalg.norm(P - P_proj, axis = 1)

	def adapt_sample(self, pts = None):
		if pts is None: return self.p - self.v, self.p + self.v
		pts = self.project(pts)
		if len(pts) == 1: return pts[0] - self.v, pts[0] + self.v
		idxes = pts.argsort(axis = 0)[:, 0]
		extend = 0.1
		t = extend + 1
		p1 = pts[idxes[0]] * t - pts[idxes[-1]] * extend
		p2 = pts[idxes[-1]] * t - pts[idxes[0]] * extend
		return p1, p2


if __name__ == '__main__':
	from Core.Visualization import KaisColor


	def main():
		pts1 = np.random.random((10, 3))
		sLine = SpaceLine.init_pv([0, 0, 0], [1, 1, 1])
		pts2 = sLine.project(pts1)
		Ds = sLine.distance(pts1)
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
