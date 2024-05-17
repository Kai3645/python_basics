import numpy as np

from Core.Basic import num2str
from Core.Visualization import KaisColor
from Core.Visualization.SpaceModel.PointModel import PointModel, CIRCLE32


class LineModel(PointModel):
	def __init__(self, count: int):
		super().__init__(count)
		self.lines = []

	@classmethod
	def init_by_obj(cls, path: str):
		fr = open(path, "r")
		pts = []
		lines = []
		for str_line in fr:
			row = str_line[:-1].split(" ")
			if row[0] == "v":
				pts.append(np.asarray(row[1:], np.float64))
				continue
			if row[0] == "l":
				lines.append(np.asarray(row[1:], int) - 1)
				continue
		lm = cls(len(pts))
		lm.points[:, :] = pts
		lm.colors[:, :] = KaisColor.plotColor("white")
		lm.lines = lines
		return lm

	def save_to_obj(self, path: str):
		colors = self.colors.astype(float) / 255
		fw = open(path, "w")
		for pt, vc in zip(self.points, colors):
			fw.write("v " + num2str(pt, 4, separator = " ") + " ")
			fw.write(num2str(vc, 6, separator = " ") + "\n")
		for ln in self.lines:
			fw.write("l " + num2str(ln + 1, 0, separator = " ") + "\n")
		fw.close()

	def merge(self, other):
		length = len(self)
		self.points = np.concatenate((self.points, other.points))
		self.colors = np.concatenate((self.colors, other.colors))
		if type(other) == PointModel: return self
		for line in other.lines: self.lines.append(line + length)
		return self

	@classmethod
	def new_line(cls, vertex1, vertex2, color):
		pts1 = np.atleast_2d(vertex1)
		pts2 = np.atleast_2d(vertex2)
		length = len(pts1)
		lm = cls(length * 2)
		lm.points[:length, :] = pts1
		lm.points[length:, :] = pts2
		lm.colors[:length, :] = color
		lm.colors[length:, :] = color
		for i in range(length):
			j = length + i
			tmp = np.asarray([i, j])
			lm.lines.append(tmp)
		return lm

	@classmethod
	def new_polyline(cls, vertex, color):
		pts = np.atleast_2d(vertex)
		length = len(pts)
		if length <= 1: return None
		lm = cls(length)
		lm.points[:, :] = pts
		lm.colors[:, :] = color
		lm.lines.append(np.arange(length))
		return lm

	def copy(self):
		lm = self.__class__(len(self))
		lm.points[:, :] = self.points
		lm.colors[:, :] = self.colors
		lm.lines = self.lines.copy()
		return lm

	@classmethod
	def prefab_wireCamera(cls, w, h, f, color):
		pts = np.asarray([
			[0, 0, 0], [1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1],
		], np.float64)
		lm = cls(5)
		lm.points[:, 2] = pts[:, 2] * f
		lm.points[:, 0] = pts[:, 0] * (w / 2)
		lm.points[:, 1] = pts[:, 1] * (h / 2)
		lm.colors[:, :] = color
		lm.lines = [np.asarray([1, 2, 0, 1, 2, 3, 4, 0, 3, 4, 1])]
		return lm

	@classmethod
	def prefab_ellipse(cls, a, b, color):
		vertex2D = np.asarray(CIRCLE32)
		length = len(vertex2D)
		lm = cls(length)
		lm.points[:, 0] = vertex2D[:, 0] * a
		lm.points[:, 1] = vertex2D[:, 1] * b
		lm.colors[:, :] = color
		line = np.arange(length + 1)
		line[-1] = 0
		lm.lines = [line]
		return lm

	@classmethod
	def new_mesh(cls, mesh, color):
		pts = np.atleast_2d(mesh.points())
		ev_table = mesh.edge_vertex_indices()
		lm = cls(len(pts))
		lm.points[:, :] = pts
		lm.colors[:, :] = color
		lm.lines = [idxes for idxes in ev_table]
		return lm


if __name__ == '__main__':
	def main():
		pass


	main()
	pass
