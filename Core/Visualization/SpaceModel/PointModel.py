import laspy
import numpy as np
from scipy.stats import multivariate_normal

from Core.Basic import errPrint, num2str
from Core.Visualization import KaisColor

CIRCLE16 = [  # top view CCW
	[1, 0], [886 / 959, 305 / 797], [408 / 577, 408 / 577], [305 / 797, 886 / 959],
	[0, 1], [-305 / 797, 886 / 959], [-408 / 577, 408 / 577], [-886 / 959, 305 / 797],
	[-1, 0], [-886 / 959, -305 / 797], [-408 / 577, -408 / 577], [-305 / 797, -886 / 959],
	[0, -1], [305 / 797, -886 / 959], [408 / 577, -408 / 577], [886 / 959, -305 / 797],
]
CIRCLE32 = [  # top view CCW
	[1, 0], [970 / 989, 151 / 774], [886 / 959, 305 / 797], [819 / 985, 5 / 9],
	[408 / 577, 408 / 577], [5 / 9, 819 / 985], [305 / 797, 886 / 959], [151 / 774, 970 / 989],
	[0, 1], [-151 / 774, 970 / 989], [-305 / 797, 886 / 959], [-5 / 9, 819 / 985],
	[-408 / 577, 408 / 577], [-819 / 985, 5 / 9], [-886 / 959, 305 / 797], [-970 / 989, 151 / 774],
	[-1, 0], [-970 / 989, -151 / 774], [-886 / 959, -305 / 797], [-819 / 985, -5 / 9],
	[-408 / 577, -408 / 577], [-5 / 9, -819 / 985], [-305 / 797, -886 / 959], [-151 / 774, -970 / 989],
	[0, -1], [151 / 774, -970 / 989], [305 / 797, -886 / 959], [5 / 9, -819 / 985],
	[408 / 577, -408 / 577], [819 / 985, -5 / 9], [886 / 959, -305 / 797], [970 / 989, -151 / 774],
]


class PointModel:
	def __init__(self, count: int):
		self.points = np.zeros((count, 3), np.float64)
		self.colors = np.zeros((count, 3), np.uint16)

	def __len__(self):
		return len(self.points)

	@classmethod
	def init_random(cls, count: int):
		pm = cls(count)
		pm.points[:, :] = np.random.random((count, 3))
		pm.colors[:, :] = KaisColor.rand(length = count)
		return pm

	@classmethod
	def init_by_las(cls, path: str):
		fr = laspy.read(path)
		pm = cls(fr.header.point_count)
		pm.points[:, :] = fr.xyz
		if fr.point_format.id in [2, 3, 5, 7, 8, 10]:
			pm.colors[:, 0] = fr.red
			pm.colors[:, 1] = fr.green
			pm.colors[:, 2] = fr.blue
		else:
			pm.colors[:, :] = 200
		return pm

	def save_to_las(self, path: str):
		if len(self) < 1: return errPrint(">> err, empty las data .. ")

		hdr = laspy.LasHeader(version = "1.2", point_format = 2)
		hdr.scales = (0.001, 0.001, 0.001)
		fw = laspy.LasData(hdr)

		fw.x = self.points[:, 0]
		fw.y = self.points[:, 1]
		fw.z = self.points[:, 2]

		fw.red = self.colors[:, 0]
		fw.green = self.colors[:, 1]
		fw.blue = self.colors[:, 2]

		fw.write(path)

	@classmethod
	def init_by_csv(cls, path: str):
		# todo: update when needed
		pass

	def save_to_csv(self, path: str):
		fw = open(path, "w")
		fw.write("x,y,z,red,green,blue,intensity\n")
		for i in range(len(self)):
			tmp_str = num2str(self.points[i], 4, separator = ",") + ","
			tmp_str += num2str(self.colors[i], 0, separator = ",") + "\n"
			fw.write(tmp_str)
		fw.close()

	def merge(self, other):
		self.points = np.concatenate((self.points, other.points))
		self.colors = np.concatenate((self.colors, other.colors))
		return self

	@classmethod
	def new_point(cls, point, color):
		points = np.atleast_2d(point)
		pm = cls(len(points))
		pm.points[:, :] = points
		pm.colors[:, :] = color
		return pm

	def copy(self):
		pm = self.__class__(len(self))
		pm.points[:, :] = self.points
		pm.colors[:, :] = self.colors
		return pm

	def trans(self, R = None, T = None):
		if R is not None:
			self.points[:, :] = self.points.dot(np.transpose(R))
		if T is not None:
			self.points[:, :] = self.points + T
		return self

	def transCopy(self, R = None, T = None):
		pm = self.copy()
		return pm.trans(R, T)

	@classmethod
	def new_normalCloud(cls, mean, covar, color, total):
		normal = multivariate_normal(mean, covar)
		return cls.new_point(normal.rvs(total), color)

	@classmethod
	def new_mesh(cls, mesh, color):
		return cls.new_point(mesh.points(), color)


if __name__ == '__main__':
	def main():
		pass


	main()
	pass
