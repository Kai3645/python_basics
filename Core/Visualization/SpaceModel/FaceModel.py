import math

import numpy as np

from Core.Basic import num2str
from Core.Visualization import KaisColor
from Core.Visualization.SpaceModel.PointModel import PointModel, CIRCLE16


class FaceModel(PointModel):
	def __init__(self, count: int):
		super().__init__(count)
		self.faces = []

	def save_to_ply(self, path: str):
		face_count = sum([len(face) - 2 for face in self.faces])
		str_heads = [
			"ply",
			"format ascii 1.0",
			f"element vertex {len(self)}",
			"property double x",
			"property double y",
			"property double z",
			"property uchar red",
			"property uchar green",
			"property uchar blue",
			f"element face {face_count}",
			"property list uint int vertex_index",
			"end_header"
		]
		fw = open(path, "w")
		fw.write("\n".join(str_heads) + "\n")
		for pt, fc in zip(self.points, self.colors):
			fw.write(num2str(pt, 4, separator = " ") + " ")
			fw.write(num2str(fc, 0, separator = " ") + "\n")
		for face in self.faces:
			for f_i, f_j in zip(face[1:-1], face[2:]):
				fw.write(f"3 {face[0]} {f_i} {f_j}\n")
		fw.close()

	def merge(self, other):
		length = len(self)
		self.points = np.concatenate((self.points, other.points))
		self.colors = np.concatenate((self.colors, other.colors))
		if type(other) == PointModel: return self
		for face in other.faces: self.faces.append(face + length)
		return self

	@classmethod
	def new_face(cls, vertex, color):
		"""
		convex polygon required
		:param vertex:
		:param color:
		:return:
		"""
		pts = np.atleast_2d(vertex)
		length = len(pts)
		if length <= 2: return None

		lm = cls(length)
		lm.points[:, :] = pts
		lm.colors[:, :] = color
		for i in range(1, length - 1):
			lm.faces.append(np.asarray([0, i, i + 1], int))
		return lm

	def copy(self):
		fm = self.__class__(len(self))
		fm.points[:, :] = self.points
		fm.colors[:, :] = self.colors
		fm.faces = self.faces.copy()
		return fm

	@classmethod
	def prefab_arrow(cls, start, end, r, color):
		"""
		:param start: (3, ) array like
		:param end: (3, ) array like
		:param r: radius of arrow
		:param color:
		:return:
		"""
		v = np.array([end[0] - start[0], end[1] - start[1], end[2] - start[2]], np.float64)
		length = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
		v /= length

		vertex2D = np.asarray(CIRCLE16) * r
		n = len(vertex2D)
		n2 = n * 2
		n3 = n * 3
		model = cls(n3 + 1)
		model.points[:n, 1:] = vertex2D
		model.points[n:n2, 1:] = vertex2D
		model.points[n:n2, 0] = length - 20 * r
		model.points[n2:n3, 1:] = vertex2D * 2.5
		model.points[n2:n3, 0] = length - 16 * r
		model.points[n3, 0] = length
		model.colors[:, :] = color
		for i in range(n):
			j = (i + 1) % n
			k = j + n
			model.faces.append(np.asarray([j, j + n, i + n, i], int))
			model.faces.append(np.asarray([k, k + n, i + n2, i + n], int))
			model.faces.append(np.asarray([j + n2, n3, i + n2], int))
		for i in range(1, n - 1):
			model.faces.append(np.asarray([0, i, i + 1], int))
		u, s, vh = np.linalg.svd(np.asarray([
			[v[0] * v[0], v[1] * v[0], v[2] * v[0]],
			[v[0] * v[1], v[1] * v[1], v[2] * v[1]],
			[v[0] * v[2], v[1] * v[2], v[2] * v[2]],
		]))
		return model.trans(u, start)

	@classmethod
	def prefab_axis(cls, r, scales, cnames):
		colors = KaisColor.plotColor(cnames)
		fm = cls(0)
		fm.merge(cls.prefab_arrow((0, 0, 0), (scales[0], 0, 0), r, color = colors[0]))
		fm.merge(cls.prefab_arrow((0, 0, 0), (0, scales[1], 0), r, color = colors[1]))
		fm.merge(cls.prefab_arrow((0, 0, 0), (0, 0, scales[2]), r, color = colors[2]))
		return fm

	@classmethod
	def new_mesh(cls, mesh, color):
		pts = np.atleast_2d(mesh.points())
		fv_table = mesh.face_vertex_indices()
		fm = cls(len(pts))
		fm.points[:, :] = pts
		fm.colors[:, :] = color
		fm.faces = [idxes[idxes >= 0] for idxes in fv_table]
		return fm


if __name__ == '__main__':
	def main():
		pass


	main()
	pass
