import math

import numpy as np

from Core.Basic import NUM_ERROR, KaisLog
from Core.Geometry import Rotation_AxisAngle, AxisAngle_Rotation

log = KaisLog.get_log()
SAMPLE_RANGE_MIN = 2000  # [frame] cover at least 2 turns(left + right)
DAMPING = 0.7
DAMPING_ = 1 - DAMPING
NUM_CD = 60


class InitPoseEstimator:
	"""
	estimate camera initiate pose on vehicle, solve based on RANSAC
	axis_x: vehicle x in camera coordSys
	axis_y: vehicle y in camera coordSys
	axis_z: solve by axis_x cross axis_y

	P_a: (n, 3) array like, points in Any(_a) cs
	tP_v = R_v2c * tP_c + tT_v2c
	P_v = P_c * tR_v2c + T_v2c <---- funny
	P_v = P_c * R_c2v + T_v2c  <----  not funny
	"""

	def __init__(self, sr: float = SAMPLE_RANGE_MIN, dtype = np.float32):
		sr = int(sr)
		self.sr = max(SAMPLE_RANGE_MIN, sr)
		self.init_cd = NUM_CD
		self.is_good = False
		# memo: get R, t from frames
		#   xs: t
		#   ys: R-axis
		#   as: R-angle
		self.sample_xs = np.zeros((sr, 3), dtype)
		self.sample_ys = np.zeros((sr, 3), dtype)
		self.sample_as = np.zeros(sr, dtype)
		self.curr_ix = 0
		self.curr_iy = 0
		self.is_xs_full = False
		self.is_ys_full = False
		self.inlier_xs = None
		self.inlier_ys = None
		self.axis_x = np.asarray([0, 0, 1], dtype)
		self.axis_y = np.asarray([0, -1, 0], dtype)
		self.axis_z = np.asarray([1, 0, 0], dtype)
		pass

	def __str__(self):
		"""
		format: seperator ','
			R_v2c: str[0:9], '.9f'
			is_good: str[9], '1d'
				0: False
				1: True
			len(xs): str[10], 'd'
			len(ys): str[11], 'd'
		:return:
		"""
		tmp = self.get_R_v2c().ravel().round(9)
		tmp_str = ",".join(tmp)
		if self.is_good: tmp_str += ",1"
		else: tmp_str += ",0"
		if self.is_xs_full: tmp_str += f",{self.sr}"
		else: tmp_str += f",{self.curr_ix}"
		if self.is_ys_full: tmp_str += f",{self.sr}"
		else: tmp_str += f",{self.curr_iy}"
		return tmp_str

	def get_sample_xs(self):
		if self.is_xs_full: return self.sample_xs
		return self.sample_xs[:self.curr_ix]

	def _add_sample_x(self, x):
		self.sample_xs[self.curr_ix, :] = x
		self.curr_ix += 1
		if self.curr_ix >= self.sr:
			self.is_xs_full = True
			self.curr_ix = 0
		return self.get_sample_xs()

	def get_sample_ys(self):
		if self.is_ys_full: return self.sample_ys, self.sample_as
		return self.sample_ys[:self.curr_iy], self.sample_as[:self.curr_iy]

	def _add_sample_y(self, y, a):
		self.sample_ys[self.curr_iy, :] = y
		self.sample_as[self.curr_iy] = a
		self.curr_iy += 1
		if self.curr_iy >= self.sr:
			self.is_ys_full = True
			self.curr_iy = 0
		return self.get_sample_ys()

	def get_nx(self):
		if self.is_xs_full: return self.sr
		return self.curr_ix

	def get_ny(self):
		if self.is_ys_full: return self.sr
		return self.curr_iy

	def _update_x(self, T):
		"""
		:param T:
		:return:
			flag: if changed
		"""
		X = self._add_sample_x(T)
		nx = self.get_nx()
		Q2i = max(1, nx // 2)
		# Q3i = max(1, int(nx / 4 * 3))
		# --------- --------- --------- ---------
		H = self.axis_y.dot(X.T)
		idxes = np.argsort(np.abs(H))
		H_ = H[idxes[:Q2i]]
		# H_ = H[idxes[:Q3i]]
		sigma_H = math.sqrt(H_.dot(H_.T) / Q2i)
		# sigma_H = math.sqrt(H_.dot(H_.T) / Q3i)
		H /= max(sigma_H, 0.05)
		# --------- --------- --------- ---------
		Lx = self.axis_z.dot(X.T)
		Ly = np.abs(self.axis_x.dot(X.T))
		A = np.arctan2(Lx, Ly)
		idxes = np.argsort(np.abs(A))
		A_ = A[idxes[:Q2i]]
		sigma_A = math.sqrt(A_.dot(A_.T) / Q2i)
		# A_ = A[idxes[:Q3i]]
		# sigma_A = math.sqrt(A_.dot(A_.T) / Q3i)
		A /= max(sigma_A, 0.1)
		# --------- --------- --------- ---------
		U = np.asarray([A, H]).T
		idxes = None
		idx_best = nx
		mean = np.mean(U, axis = 0)
		for i in range(20):
			D = np.linalg.norm(U - mean, axis = 1)
			idxes = np.argsort(D)
			if idxes[0] == idx_best: break
			idx_best = idxes[0]
			mean = np.mean(U[idxes[:Q2i]], axis = 0)
			# mean = np.mean(U[idxes[:Q3i]], axis = 0)
		self.inlier_xs = idxes[:Q2i]
		# self.inlier_xs = idxes[:Q3i]
		x_new = X[idx_best] / np.linalg.norm(X[idx_best])
		if x_new[2] < 0: x_new = -x_new
		# --------- --------- --------- ---------
		# canvas2.histogram(H, fit_on = True)
		# canvas2.histogram(A, fit_on = True)
		# canvas2.ax.scatter(A, H, color = "white", marker = ".", s = 10)
		# canvas2.set_axis(equal_axis = True, xlim = (-1.1, 1.1))
		# canvas2.show(True, 0.1)
		# canvas2.clear()
		# --------- --------- --------- ---------
		x_new = x_new * DAMPING_ + self.axis_x * DAMPING
		x_new = x_new / np.linalg.norm(x_new)
		delta = 1 - x_new.dot(self.axis_x)
		log.info(f"x-axis = ({x_new[0]:.3f}, {x_new[1]:.3f}, {x_new[2]:.3f}), delta = {delta:.4e} ..")
		self.axis_x[:] = x_new
		return delta > 2e-4  # about 1 [deg]

	def _update_y(self, v, a):
		"""
		goal:
			PCA_vector_of_X * axis_y = 0
			axis_x * axis_y = 0
			PCA_vector_of_Y ~ axis_y
		:param v:
		:param a:
		:return:
			flag: if changed
		"""
		if a < NUM_ERROR or a > math.pi / 3: return False
		Y, A = self._add_sample_y(v * a, a)
		ny = self.get_ny()
		# --------- --------- --------- ---------
		# Oct7i = max(1, int(ny / 8 * 7))
		# A_ = A[np.argsort(A)[:Oct7i]]
		# sigma_A = math.sqrt(A_.dot(A_) / Oct7i)
		# valid_A = A < max(sigma_A * 4, 0.1)
		# --------- --------- --------- ---------
		i_a = max(0, int(ny * 0.96))
		A_th = A[np.argsort(A)[i_a]]
		valid_A = A < max(A_th / 3 * 4, 0.01)
		# --------- --------- --------- ---------
		Q3i = max(1, int(ny / 4 * 3))
		K = self.axis_y.dot(Y.T)
		H = Y - np.tile(K, (3, 1)).T * self.axis_y
		D = np.linalg.norm(H, axis = 1)
		D_ = D[np.argsort(D)[:Q3i]]
		sigma_D = math.sqrt(D_.dot(D_) / Q3i)
		valid_D = D < max(sigma_D * 3, 0.01)
		# --------- --------- --------- ---------
		valid = np.logical_and(valid_A, valid_D)
		self.inlier_ys = np.where(valid)[0]
		Y = Y[self.inlier_ys]
		Uz = self.axis_z.dot(Y.T)
		Uy = self.axis_y.dot(Y.T)
		# --------- --------- --------- ---------
		den = max(1, len(Y) - 1)
		V = np.zeros((2, 2))
		V[0, 0] = Uz.dot(Uz) / den
		V[0, 1] = Uz.dot(Uy) / den
		V[1, 0] = V[0, 1]
		V[1, 1] = Uy.dot(Uy) / den
		# --------- --------- --------- ---------
		u, s, vh = np.linalg.svd(V)
		y_new = self.axis_z * u[0, 0] + self.axis_y * u[1, 0]
		if y_new[1] > 0: y_new = -y_new
		y_new = y_new * DAMPING_ + self.axis_y * DAMPING
		y_new -= self.axis_x.dot(y_new) * self.axis_x
		y_new /= np.linalg.norm(y_new)
		# --------- --------- --------- ---------
		# canvas2.ax.scatter(Ux[1] - 0.25, Ux[0], color = "white", marker = ".", s = 10)
		# canvas2.ax.scatter(Uy[1] + 0.25, Uy[0], color = "orange", marker = ".", s = 10)
		# canvas2.set_axis(equal_axis = True, xlim = (-0.7, 0.7))
		# canvas2.show(True, 0.01)
		# canvas2.clear()
		# --------- --------- --------- ---------
		delta = 1 - y_new.dot(self.axis_y)
		log.info(f"y-axis = ({y_new[0]:.3f}, {y_new[1]:.3f}, {y_new[2]:.3f}), delta = {delta:.2e} ..")
		self.axis_y[:] = y_new
		return delta > 2e-4  # about 1 [deg]

	def update(self, R, T):
		"""
		:param R:
		:param T: should be normalized
		:return:
			flag: if is_good
		"""
		v, a = Rotation_AxisAngle(R)
		T = T.dot(AxisAngle_Rotation(v, -a / 2))
		flag_x = self._update_x(T)
		flag_y = self._update_y(v, a)
		self.axis_z[:] = np.cross(self.axis_x, self.axis_y)

		if flag_y or flag_x:
			self.is_good = False
			self.init_cd = NUM_CD
			return False
		self.init_cd -= 1
		if self.init_cd > 0: return False
		if self.is_good: return True
		log.critical("ipe is good now")
		self.is_good = True
		return True

	def get_R_v2c(self):
		return np.asarray([self.axis_x, self.axis_y, self.axis_z])

	def get_R_c2v(self):
		return np.asarray([self.axis_x, self.axis_y, self.axis_z]).T


if __name__ == '__main__':
	from Core.Basic import num2str
	from Core.Geometry import SpacePlane, SO3_Rotation
	from Core.Visualization import KaisCanvas, SpaceCanvas, KaisColor
	from Core.SpaceDynamic import DCoordSys

	folder_out = "/home/kai/PycharmProjects/pyCenter/d_2022_0828/out/"

	KaisLog.set_level("info")
	canvas2 = KaisCanvas()
	sCanvas = SpaceCanvas(folder_out + "pe")


	def main():
		# ==================== read sample points ==============================
		total = 0
		X, Y = [], []
		fr = open(folder_out + "axis.txt", "r")
		for line in fr:
			row = np.asarray(line[:-1].split(","), np.float64)
			X.append(row[3:])
			Y.append(row[:3])
			total += 1
		fr.close()
		X = np.asarray(X)
		Y = np.asarray(Y)

		# ==================== update ==============================
		data = []
		ipe = InitPoseEstimator(1e4)
		trigger = True
		for i in np.arange(total):
			log.info(f"{i}, {ipe.init_cd}")

			# pretend got R, T from images
			R = SO3_Rotation(Y[i])
			is_good = ipe.update(R, X[i])

			if trigger and is_good:
				log.critical(f"ipe is good at {i}")
				trigger = False

			R_v2c = ipe.get_R_v2c()
			if R_v2c is None: continue

			# solve Euler rotation from vehicle to camera
			data.append(DCoordSys.deconvV2C(R_v2c))

		# ==================== visualize ==============================
		data = np.rad2deg(data)
		# data = np.asarray(data)
		start, end = 0, len(data)
		# start = 140
		# end = 400
		canvas2.ax.plot(data[start:end, 0], label = "roll")
		canvas2.ax.plot(data[start:end, 1], label = "pitch")
		canvas2.ax.plot(data[start:end, 2], label = "yaw")
		canvas2.set_axis(equal_axis = False, legend_on = True)
		canvas2.show()
		canvas2.save(folder_out + "rotation.jpg")
		canvas2.clear()
		# --------- --------- --------- ---------
		X = ipe.get_sample_xs()
		valid = np.full(len(X), False)
		valid[ipe.inlier_xs] = True
		invalid = np.logical_not(valid)
		sCanvas.add_point("x_in", X[valid], color = KaisColor.plotColorMap("cool", int(np.sum(valid))))
		sCanvas.add_point("x_out", X[invalid], color = KaisColor.plotColorMap("cool", int(np.sum(invalid))))
		valid[:] = False
		Y, _ = ipe.get_sample_ys()
		Y = Y * 4
		valid = np.full(len(Y), False)
		valid[ipe.inlier_ys] = True
		invalid = np.logical_not(valid)
		sCanvas.add_point("y_in", Y[valid], color = KaisColor.plotColorMap("spring", int(np.sum(valid))))
		sCanvas.add_point("y_out", Y[invalid], color = KaisColor.plotColorMap("spring", int(np.sum(invalid))))
		sCanvas.add_line("axis_y", ipe.axis_y * -0.8, ipe.axis_y * 1.6)
		sCanvas.add_line("axis_x", ipe.axis_x * -0.8, ipe.axis_x * 1.6)
		# --------- --------- --------- ---------
		sCanvas.add_axis("axis", R = ipe.get_R_c2v(), r = 0.01)
		sCanvas.add_wireCamera("camera", s = 0.6)
		# --------- --------- --------- ---------
		sPlane = SpacePlane.init_pn((0, 0, 0), ipe.axis_y)
		p1, p2 = sPlane.adapt_sample()
		sCanvas.add_line("plane", p1 * 1.25, p2 * 1.25)
		# --------- --------- --------- ---------
		fw = open(folder_out + "axis_result.txt", "w")
		fw.write("axis-x," + num2str(ipe.axis_x, 10, separator = ",") + "\n")
		fw.write("axis-y," + num2str(ipe.axis_y, 10, separator = ",") + "\n")
		fw.close()
		# --------- --------- --------- ---------
		H = sPlane.distance(X)
		Lx = ipe.axis_z.dot(X.T)
		Ly = np.abs(ipe.axis_x.dot(X.T))
		A = np.arctan2(Lx, Ly) / np.pi * 2
		Pa = np.asarray([A, H], np.float64).T
		Pb = np.asarray([Lx, H], np.float64).T
		# --------- --------- --------- ---------
		canvas2.draw_points(Pb, color = "orangered", marker = ".", s = 10)
		canvas2.draw_points(Pa, color = "white", marker = ".", s = 10)
		canvas2.set_axis(equal_axis = True, xlim = (-1.1, 1.1))
		canvas2.save(folder_out + "axis_x.jpg")
		canvas2.show()
		canvas2.clear()
		# --------- --------- --------- ---------
		canvas2.histogram(A, denoise_on = True)
		canvas2.set_axis(equal_axis = False, xlim = (-0.25, 0.25))
		canvas2.save(folder_out + "axis_x_a.jpg")
		canvas2.show()
		canvas2.clear()
		# --------- --------- --------- ---------
		canvas2.histogram(H, denoise_on = True)
		canvas2.set_axis(equal_axis = False, xlim = (-0.2, 0.2))
		canvas2.save(folder_out + "axis_x_b.jpg")
		canvas2.show()
		canvas2.clear()
		# --------- --------- --------- ---------
		W = np.asarray([ipe.axis_x, ipe.axis_z], np.float64)
		inv_WW = np.linalg.inv(W.dot(W.T))
		P = Y.dot(W.T).dot(inv_WW)
		# --------- --------- --------- ---------
		canvas2.draw_points(P, color = "white", marker = ".", s = 10)
		canvas2.set_axis(equal_axis = False, xlim = (-0.6, 0.6), ylim = (-0.5, 0.5))
		canvas2.save(folder_out + "axis_y.jpg")
		canvas2.show()
		canvas2.clear()
		# --------- --------- --------- ---------
		canvas2.histogram(P[:, 0], denoise_on = True)
		canvas2.set_axis(equal_axis = False, xlim = (-0.1, 0.1))
		canvas2.save(folder_out + "axis_y_a.jpg")
		canvas2.show()
		canvas2.clear()
		# --------- --------- --------- ---------
		canvas2.histogram(P[:, 1], denoise_on = True)
		canvas2.set_axis(equal_axis = False, xlim = (-0.1, 0.1))
		canvas2.save(folder_out + "axis_y_b.jpg")
		canvas2.show()
		canvas2.clear()

		pass


	main()

	canvas2.close()
	sCanvas.save()
	pass
