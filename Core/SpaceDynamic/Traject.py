import numpy as np
from tqdm import tqdm

from Core.Basic import file_length, timestamp2str, str2timestamp
from Core.Basic import num2str, argSample, argIntercept, argInterpolate, scaling
from Core.Geometry import Euler_Interpolate
from Core.Visualization import KaisLas, KaisColor


class Traject:
	"""
	_center: (cx, cy, cz) m, float(Integer)
	_initialTime: sec, utc timestamp, float(Integer)
	---------
	time: sec
	pos: (x, y, z) m
	rot: (roll, pitch, yaw) radians
	----------------------------------------
	csv_format: [time], [pos], [rot]
	----------------------------------------
	insure data accuracy
	time: .6f
	pos: .4f
	rot: .8f[rad], .6f[deg]
	"""

	def __init__(self, count: int):
		self._center = np.zeros(3)
		self._initialTime = 0.0
		self.Time = np.zeros(count)
		self.Pos = np.zeros((count, 3))
		self.Rot = np.zeros((count, 3))

	def __len__(self):
		return len(self.Time)

	def length(self):
		return len(self)

	def range(self, start: int = 0):
		return range(start, self.length())

	def time_at(self, idx: int):
		return self.Time[idx] + self._initialTime

	def pos_at(self, idx: int):
		return self.Pos[idx] + self._center

	def rot_at(self, idx: int, toDeg: bool = False):
		if toDeg: return np.rad2deg(self.Rot[idx])
		return self.Rot[idx]

	def data_at(self, idx: int):
		time = self.time_at(idx)
		pos = self.pos_at(idx)
		rot = self.rot_at(idx)
		return time, pos, rot

	def str_at(self, idx: int, separator = ", "):
		tmp_str = f"{self.Time[idx]:.4f}" + separator
		tmp_str += num2str(self.Pos[idx], 4, separator = separator) + separator
		tmp_str += num2str(self.rot_at(idx, True), 6, separator = separator)
		return tmp_str

	def set_time_at(self, idx: int, time: float):
		self.Time[idx] = time - self._initialTime
		return self

	def set_pos_at(self, idx: int, pos):
		self.Pos[idx, :] = pos - self._center
		return self

	def set_rot_at(self, idx: int, rot):
		self.Rot[idx, :] = rot
		return self

	def set_center(self, center):
		self._center[:] = center
		return self

	def set_initialTime(self, initialTime):
		self._initialTime = initialTime
		return self

	def get_center(self):
		return np.copy(self._center)

	def swap_center(self, new_center):
		self.Pos += self._center - new_center
		self._center[:] = new_center
		return self

	def update_center(self):
		delta_center = np.median(self.Pos, axis = 0).round()
		self.Pos -= delta_center
		self._center += delta_center
		self.Pos = np.round(self.Pos, 4)
		return self

	def get_initialTime(self):
		return self._initialTime

	def swap_initialTime(self, new_time):
		self.Time += self._initialTime - new_time
		self._initialTime = new_time
		return self

	def update_initialTime(self):
		delta_time = np.floor(self.Time[0])
		self.Time -= delta_time
		self._initialTime += delta_time
		self.Time = np.round(self.Time, 6)
		return self

	def update_all(self):
		self.update_center()
		self.update_initialTime()
		return self

	def __str__(self):
		# todo: show more traject info
		if self.length() == 0: return "Empty traject"
		tmp_str = "Traject:\n"
		tmp_str += "---------------------------------------------------\n"
		tmp_str += f"initial time = {self._initialTime:.4f}\n"
		tmp_str += "utc time: " + timestamp2str(self._initialTime, "%Y-%m-%d %H:%M:%S.%f %z") + "\n"
		tmp_str += f"time range ({self.Time[0]:.4f}, {self.Time[-1]:.4f})\n"
		tmp_str += f"node count {self.length()}\n"
		tmp_str += f"map center = {num2str(self._center, 4)}\n"
		tmp_str += f"mileage = {None}\n"  # todo: <<----------
		tmp_str += "sample lines:\n"
		idxes1, idxes2, idxes3 = argSample(self.Time, (4, 4, 4))
		tmp_str += "---------------------------------------------------\n"
		for idx in idxes1: tmp_str += f"\t{idx},\t{self.str_at(idx)}\n"
		tmp_str += "\t- - - - - - - - - - - - - - - - - - - - - - - - -\n"
		for idx in idxes2: tmp_str += f"\t{idx},\t{self.str_at(idx)}\n"
		tmp_str += "\t- - - - - - - - - - - - - - - - - - - - - - - - -\n"
		for idx in idxes3: tmp_str += f"\t{idx},\t{self.str_at(idx)}\n"
		tmp_str += "==================================================="
		return tmp_str

	def save_to_csv(self, path: str):
		fw = open(path, "w")
		fw.write("center_x,center_y,center_z,initial_t\n")
		fw.write(num2str(self._center, 4, separator = ",") + ",")
		fw.write(timestamp2str(self._initialTime, "%Y%m%d%H%M%S%f") + "\n")
		fw.write("time,x,y,z,row,pitch,yaw\n")
		for i in self.range(): fw.write(self.str_at(i, ",") + "\n")
		fw.close()

	def save_to_las(self, path: str, cmap: str = "cool", cname: str = None, **kwargs):
		length = self.length()
		myLas = KaisLas(length)
		myLas.times[:] = self.Time + self._initialTime
		myLas.center[:] = self._center
		myLas.points[:, :] = self.Pos
		if cname is not None:
			myLas.colors[:, :] = KaisColor.plotColor(cname)
		else:
			myLas.colors[:, :] = KaisColor.plotColorMap(cmap, length)
		myLas.save_to_las(path, format_id = 3, **kwargs)

	def intercept(self, limit: tuple):
		"""
		get traject between (limit0, limit1)
		:param limit: (limit0, limit1), sec, utc time
		:return: new traject
		"""
		limit = np.asarray(limit) - self._initialTime
		idxes = argIntercept(self.Time, tuple(limit))
		traject = self.__class__(len(idxes))
		traject._center[:] = self._center
		traject._initialTime = self._initialTime
		traject.Time[:] = self.Time[idxes]
		traject.Pos[:, :] = self.Pos[idxes]
		traject.Rot[:, :] = self.Rot[idxes]
		return traject.update_all()

	def init_data_at(self, idx: int, *args):
		"""
		:param idx:
		:param args: time, pos, rot
		:return:
		"""
		self.Time[idx] = args[0]
		self.Pos[idx, :] = args[1]
		self.Rot[idx, :] = args[2]

	def init_headLine_csv(self, headLine: str):
		headRow = headLine[:-1].split(",")
		self.set_center(np.float64(headRow[:3]))
		self.set_initialTime(str2timestamp(headRow[3], "%Y%m%d%H%M%S%f"))
		return self

	@staticmethod
	def unzipLine_csv(tmp_line: str):
		row = np.float64(tmp_line[:-1].split(","))
		time = row[0]
		pos = row[1:4]
		rot = np.deg2rad(row[4:7])
		return time, pos, rot

	@classmethod
	def init_by_csv(cls, path):
		count = file_length(path) - 3
		traject = cls(count)

		fr = open(path, "r")
		next(fr)  # pass title
		traject.init_headLine_csv(next(fr))
		next(fr)  # pass title
		for i, line in enumerate(tqdm(fr, total = count, desc = ">> reading")):
			arg = cls.unzipLine_csv(line)
			traject.init_data_at(i, *arg)
		fr.close()
		return traject.update_all()

	def linear(self, Time, initialTime = 0, *, rotation_order: tuple = (0, 1, 2)):
		"""
		linear interpolation
		:param Time:
		:param initialTime:
		:param rotation_order:
		:return:
		"""
		Time = Time + (initialTime - self._initialTime)

		idxes1, idxes2 = argInterpolate(self.Time, Time)
		t = scaling(self.Time[idxes1], self.Time[idxes2], Time)
		w = np.tile(t, (3, 1)).transpose()

		traject = self.__class__(len(Time))
		traject._center[:] = self._center
		traject._initialTime = self._initialTime
		traject.Time[:] = Time
		traject.Pos[:, :] = self.Pos[idxes1, :] * (1 - w) + self.Pos[idxes2, :] * w
		traject.Rot[:, :] = Euler_Interpolate(self.Rot, idxes1, idxes2, t, order = rotation_order)

		return traject.update_all()

	def push_super(self):
		"""

		:return: Traject type object
		"""
		traject = Traject(self.length())
		traject.set_center(self._center)
		traject.set_initialTime(self._initialTime)
		traject.Time[:] = self.Time
		traject.Pos[:, :] = self.Pos
		traject.Rot[:, :] = self.Rot
		return traject

	def __copy__(self):
		traject = self.__class__(self.length())
		traject.Time[:] = self.Time
		traject.Pos[:, :] = self.Pos
		traject.Rot[:, :] = self.Rot
		traject.set_center(self._center)
		traject.set_initialTime(self._initialTime)
		return traject

# def mileage(traject, Time = None): # todo: update when needed
# 	dS = np.linalg.norm(traject.Pos[1:] - traject.Pos[:-1], axis = 1)
# 	Mileage = np.zeros(traject.length(), np.float64)
# 	for i, ds in enumerate(dS, 1):
# 		Mileage[i] = Mileage[i - 1] + ds
# 	if Time is None: return Mileage, traject.Time
# 	idxes1, idxes2 = argInterpolate(traject.Time, Time)
# 	t = scaling(traject.Time[idxes1], traject.Time[idxes2], Time)
# 	Mileage = Mileage[idxes1] * (1 - t) + Mileage[idxes2] * t
# 	return Mileage, Time


# def save_pose_by_mat(path: str, Time, Mat, center): # todo: update when needed
# 	with open(path, "w") as fw:
# 		fw.write(num2str(center, 4) + "\n")
# 		for t, m in zip(Time, Mat):
# 			pos = np.asarray(m[:3, 3]).ravel()
# 			quat = Rotation_Quaternion(m[:3, :3])
# 			fw.write(f"{t:.4f}," + num2str(pos, 4) + ",")
# 			fw.write(num2str(quat.asarray(), 8) + "\n")
# 	pass


# def read_pose_by_mat(path: str): # todo: update when needed
# 	count = file_length(path) - 1
# 	Time = np.zeros(count)
# 	Mat = np.empty(count, np.matrix)
# 	with open(path, "r") as fr:
# 		center = np.float64(next(fr).split(",")[:3])
# 		for i, line in enumerate(fr):
# 			row = np.float64(line.split(","))
# 			Time[i] = row[0]
# 			m = np.asmatrix(np.eye(4))
# 			m[:3, 3] = row[1:4].reshape((3, 1))
# 			quat = Quat.init_by_array(row[4:8])
# 			m[:3, :3] = Quaternion_Rotation(quat)
# 			Mat[i] = m
# 	return Time, Mat, center
