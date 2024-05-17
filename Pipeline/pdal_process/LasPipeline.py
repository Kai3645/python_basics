import json
from tempfile import TemporaryFile

import pdal

from Core.Basic import sysInfo


class LasPipe:
	def __init__(self):
		self.filters = []

	def __str__(self):
		with TemporaryFile("w+t") as fwt:
			json.dump(
				self.filters, fwt,
				separators = (",", ":"),
				indent = 3,
			)
			fwt.seek(0)
			tmp_str = ""
			for line in fwt.readlines():
				tmp_str += line
		tmp_str = "{\"pipeline\":\n" + tmp_str + "\n}"
		return str(tmp_str)

	def init_by_json(self, path):
		fr_json = open(path, "r")
		self.filters[:] = json.load(fr_json)[:]
		return self

	def save(self, path):
		fw = open(path, "w")
		json.dump(
			self.filters, fw,
			separators = (",", ":"),
			indent = 3,
		)
		fw.close()

	def run(self):
		print(sysInfo(">> pipe running .. "))
		pipeline = pdal.Pipeline(str(self))
		pipeline.validate()  # check if our JSON and options were good
		pipeline.loglevel = 3  # really noisy
		_ = pipeline.execute()
		_ = pipeline.arrays
		_ = pipeline.metadata
		_ = pipeline.log

	def add_reader(self, path, f_type = "las"):
		f = {
			"type": "readers." + f_type,
			"filename": path,
		}
		self.filters.append(f)

	def add_writer_las(self, path, format_id, scale, offset):
		f = {
			"type": "writers.las",
			"filename": path,
			"dataformat_id": format_id,
			"scale_x": scale[0],
			"scale_y": scale[1],
			"scale_z": scale[2],
			"offset_x": offset[0],
			"offset_y": offset[1],
			"offset_z": offset[2],
		}
		self.filters.append(f)

	def add_writer_csv(self, path, csv_format):
		f = {
			"type": "writers.text",
			"filename": path,
			"order": csv_format,
		}
		self.filters.append(f)

	def add_outlier(self, n_float, n_int, basic = False):
		f = {"type": "filters.outlier"}
		if basic:
			f["method"] = "statistical"
			f["mean_k"] = n_int
			f["multiplier"] = n_float
		else:
			f["method"] = "radius"
			f["min_k"] = n_int
			f["radius"] = n_float
		self.filters.append(f)

	def add_groupby(self):
		f = {
			"type": "filters.groupby",
			"dimension": "Classification",
		}
		self.filters.append(f)

	def add_neighborclassifier(self, k):
		f = {
			"type": "filters.neighborclassifier",
			"domain": "Classification [1：1],",
			"k": k,
		}
		self.filters.append(f)

	def add_merge(self):
		f = {"type": "filters.merge"}
		self.filters.append(f)

	def add_sample(self, radius):
		f = {
			"type": "filters.sample",
			"radius": radius,
		}
		self.filters.append(f)

	def add_cluster(self, tolerance, min_points):
		f = {
			"type": "filters.cluster",
			"tolerance": tolerance,
			"min_points": min_points,
		}
		self.filters.append(f)

	# todo: add more

	pass
