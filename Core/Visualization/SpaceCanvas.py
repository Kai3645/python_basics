import os

from Core.Basic import errPrint
from Core.Visualization.KaisColor import KaisColor
from Core.Visualization.SpaceModel import *


class SpaceCanvas:
	model_compatible = {
		"point": (  # can merge to ..
			"point", "line", "face",
		),
		"line": (  # can merge to ..
			"line",
		),
		"face": (  # can merge to ..
			"face",
		),
	}

	def __init__(self, folder: str, **kwargs):
		self.layer = dict()
		self.layer_type = dict()
		assert os.path.exists(folder)
		if folder[-1] != os.sep: folder += os.sep
		self.folder = folder

		self._save_line_pt = kwargs.get("save_line_pt", False)
		pass

	def save(self):
		for name, model in self.layer.items():
			tName = self.layer_type[name]
			if tName == "point":
				model.save_to_las(self.folder + name + ".las")
				continue
			if tName == "line":
				model.save_to_obj(self.folder + name + ".obj")
				if not self._save_line_pt: continue
				model.save_to_las(self.folder + name + ".las")
				continue
			if tName == "face":
				model.save_to_ply(self.folder + name + ".ply")
				continue
			errPrint(f">> layer \"{name}\" got unexpected type \"{tName}\"")
		pass

	def remove(self, name: str):
		if name not in self.layer.keys():
			return errPrint(f">> layer \"{name}\" not exist ..")
		self.layer.pop(name)
		self.layer_type.pop(name)

	def trans(self, name: str, R, T):
		if self.layer_type.get(name) is None:
			return errPrint(f">> layer \"{name}\" not exist ..")
		self.layer[name].trans(R, T)

	def add_point(self, name: str, point, **kwargs):
		tName = self.layer_type.get(name)
		color = kwargs.get("color", KaisColor.rand())
		model = PointModel.new_point(point, color)
		if tName is None:
			self.layer[name] = model
			self.layer_type[name] = "point"
			return True
		if tName in self.model_compatible["point"]:
			self.layer[name].merge(model)
			return True
		errPrint(f">> layer \"{name}\" not compatible with point ..")
		return False

	def add_line(self, name: str, vertex1, vertex2, **kwargs):
		tName = self.layer_type.get(name)
		color = kwargs.get("color", KaisColor.rand())
		model = LineModel.new_line(vertex1, vertex2, color)
		if tName is None:
			self.layer[name] = model
			self.layer_type[name] = "line"
			return True
		if tName in self.model_compatible["line"]:
			self.layer[name].merge(model)
			return True
		errPrint(f">> layer \"{name}\" not compatible with line ..")
		return False

	def add_polyline(self, name: str, vertex, **kwargs):
		tName = self.layer_type.get(name)
		color = kwargs.get("color", KaisColor.rand())
		model = LineModel.new_polyline(vertex, color)
		if tName is None:
			self.layer[name] = model
			self.layer_type[name] = "line"
			return True
		if tName in self.model_compatible["line"]:
			self.layer[name].merge(model)
			return True
		errPrint(f">> layer \"{name}\" not compatible with line ..")
		return False

	def add_face(self, name: str, vertex, **kwargs):
		tName = self.layer_type.get(name)
		color = kwargs.get("color", KaisColor.rand())
		model = FaceModel.new_face(vertex, color)
		if tName is None:
			self.layer[name] = model
			self.layer_type[name] = "face"
			return True
		if tName in self.model_compatible["face"]:
			self.layer[name].merge(model)
			return True
		errPrint(f">> layer \"{name}\" not compatible with face ..")
		return False

	def add_normalCloud(self, name: str, mean, covar, **kwargs):
		total = kwargs.get("total", 1000)
		tName = self.layer_type.get(name)
		color = kwargs.get("color", KaisColor.rand())
		model = PointModel.new_normalCloud(mean, covar, color, total)
		if tName is None:
			self.layer[name] = model
			self.layer_type[name] = "point"
			return True
		if tName in self.model_compatible["point"]:
			self.layer[name].merge(model)
			return True
		errPrint(f">> layer \"{name}\" not compatible with point ..")
		return False

	def add_arrow(self, name: str, start, end, **kwargs):
		r = kwargs.get("r", 0.005)
		tName = self.layer_type.get(name)
		color = kwargs.get("color", KaisColor.rand())
		model = FaceModel.prefab_arrow(start, end, r, color)
		if tName is None:
			self.layer[name] = model
			self.layer_type[name] = "face"
			return True
		if tName in self.model_compatible["face"]:
			self.layer[name].merge(model)
			return True
		errPrint(f">> layer \"{name}\" not compatible with face ..")
		return False

	def add_wireCamera(self, name: str, *, R = None, T = None, **kwargs):
		s = kwargs.get("s", 1.0)
		w = kwargs.get("w", 1.6) * s
		h = kwargs.get("h", 1.0) * s
		f = kwargs.get("f", 0.8) * s
		tName = self.layer_type.get(name)
		color = kwargs.get("color", KaisColor.rand())
		model = LineModel.prefab_wireCamera(w, h, f, color)
		if tName is None:
			self.layer[name] = model.trans(R, T)
			self.layer_type[name] = "line"
			return True
		if tName in self.model_compatible["line"]:
			self.layer[name].merge(model.trans(R, T))
			return True
		errPrint(f">> layer \"{name}\" not compatible with line ..")
		return False

	def add_ellipse(self, name: str, *, R = None, T = None, **kwargs):
		a = kwargs.get("a", 1.0)
		b = kwargs.get("b", 0.6)
		tName = self.layer_type.get(name)
		color = kwargs.get("color", KaisColor.rand())
		model = LineModel.prefab_ellipse(a, b, color)
		if tName is None:
			self.layer[name] = model.trans(R, T)
			self.layer_type[name] = "line"
			return True
		if tName in self.model_compatible["line"]:
			self.layer[name].merge(model.trans(R, T))
			return True
		errPrint(f">> layer \"{name}\" not compatible with line ..")
		return False

	def add_axis(self, name: str, *, R = None, T = None, **kwargs):
		r = kwargs.get("r", 0.012)
		scales = kwargs.get("scales", (1.5, 1.2, 1))
		cnames = kwargs.get("cnames", KaisColor.axis_cnames)
		tName = self.layer_type.get(name)
		model = FaceModel.prefab_axis(r, scales, cnames)
		if tName is None:
			self.layer[name] = model.trans(R, T)
			self.layer_type[name] = "face"
			return True
		if tName in self.model_compatible["face"]:
			self.layer[name].merge(model.trans(R, T))
			return True
		errPrint(f">> layer \"{name}\" not compatible with face ..")
		return False

	def add_wireMesh(self, name: str, mesh, *, R = None, T = None, **kwargs):
		tName = self.layer_type.get(name)
		color = kwargs.get("color", KaisColor.rand())
		model = LineModel.new_mesh(mesh, color)
		if tName is None:
			self.layer[name] = model.trans(R, T)
			self.layer_type[name] = "line"
			return True
		if tName in self.model_compatible["line"]:
			self.layer[name].merge(model.trans(R, T))
			return True
		errPrint(f">> layer \"{name}\" not compatible with line ..")

	def add_mesh(self, name: str, mesh, *, R = None, T = None, **kwargs):
		tName = self.layer_type.get(name)
		color = kwargs.get("color", KaisColor.rand())
		model = FaceModel.new_mesh(mesh, color)
		if tName is None:
			self.layer[name] = model.trans(R, T)
			self.layer_type[name] = "face"
			return True
		if tName in self.model_compatible["face"]:
			self.layer[name].merge(model.trans(R, T))
			return True
		errPrint(f">> layer \"{name}\" not compatible with face ..")


if __name__ == '__main__':
	import math
	import numpy as np
	from Core.Geometry import CoordSys
	from Core.Visualization.SpaceModel.PointModel import CIRCLE32


	def main():
		folder = "/home/kai/PycharmProjects/pyCenter/_m_2022_07/d_2022_0720"
		canvas = SpaceCanvas(folder)

		total = 100
		points = np.random.random((total, 3)) - 0.5
		canvas.add_point("sample_points", points, color = KaisColor.rand(length = total))

		total = 20
		vertex1 = np.random.random((total, 3)) - 0.5
		vertex1[:, 1] += 1
		vertex2 = np.copy(vertex1)
		vertex2[:, 1] += 1
		canvas.add_line("sample_lines", vertex1, vertex2, color = KaisColor.plotColor("red"))

		total = 20
		points = np.random.random((total, 3)) - 0.5
		points[:, 0] -= 1
		canvas.add_polyline("sample_lines", points)

		total = len(CIRCLE32)
		points = np.zeros((total, 3))
		points[:, [0, 2]] = CIRCLE32
		points[:, 0] *= 0.5
		points[:, 2] *= 0.3
		points[:, 0] += 1.1
		points[:, 1] += 1.2
		canvas.add_face("sample_faces", points)

		total = 2000
		mean = (1.1, 0.5, 0)
		covar = [[0.0278, 0, 0], [0, 0.36, 0], [0, 0, 0.01]]
		canvas.add_normalCloud("prefab_points", mean, covar,
		                       color = KaisColor.rand(length = total),
		                       total = total)
		canvas.add_wireCamera("prefab_lines",
		                      R = CoordSys.conv((-math.pi / 2, 0, 0)), T = (-1, 1, 0),
		                      w = 0.8, h = 0.6, f = 1)
		canvas.add_ellipse("prefab_lines",
		                   R = CoordSys.conv((0, 0, math.pi / 2)), T = (1.1, 0.5, 0),
		                   a = 1.8, b = 0.5)
		canvas.add_arrow("prefab_faces", (2, -0.7, 0), (2, 1.8, 0),
		                 color = KaisColor.plotColor("gold"), r = 0.01)
		canvas.add_axis("prefab_faces", scales = (4.2, 3.6, 1.5),
		                cnames = KaisColor.axis_cnames, r = 0.012,
		                T = (-1.8, -1.0, -0.7))

		canvas.save()
		pass


	main()
	pass
