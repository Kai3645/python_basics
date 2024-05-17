import numpy as np

from Core.Basic import NUM_ZERO, KaisLog

log = KaisLog.get_log()


class CVCamera:
	def __init__(self, para, frame_size, R = None, T = None):
		"""
		:param para: (fx, fy, cx, cy)
		:param frame_size: (w, h)
		:param R: (3, 3) array, rotation matrix
		:param T: (3,) array, transition
		"""
		self.fx = para[0]
		self.fy = para[1]
		self.cx = para[2]
		self.cy = para[3]
		self.frame_w = frame_size[0]
		self.frame_h = frame_size[1]
		self.roi = (-self.cx, -self.cy, self.frame_w - self.cx, self.frame_h - self.cy)
		# camera pose in global
		self.R = np.eye(3)
		if R is not None: self.R[:, :] = R
		self.tR = self.R.T  # same as inv_R
		self.T = np.zeros(3)
		if T is not None: self.T[:] = np.ravel(T)
		pass

	def camera_para(self):
		return self.fx, self.fy, self.cx, self.cy

	def frame_size(self):
		return self.frame_w, self.frame_h

	def __str__(self):
		tmp_str = "camera >> ("
		tmp_str += ", ".join(self.camera_para()) + "), ( "
		tmp_str += ", ".join(self.frame_size()) + " )"
		return tmp_str

	def getMat(self):
		return np.asarray([
			[self.fx, 0, self.cx],
			[0, self.fy, self.cy],
			[0, 0, 1],
		], np.float64)

	def getMat_inv(self):
		return np.asarray([
			[1 / self.fx, 0, -self.cx / self.fx],
			[0, 1 / self.fy, -self.cy / self.fy],
			[0, 0, 1],
		], np.float64)

	def duplicate(self, R = None, T = None):
		para = self.camera_para()
		size = self.frame_size()
		return self.__class__(para, size, R, T)

	def projection(self, P, z_near: float = 0.5, z_far: float = 50, force_in_frame: bool = False):
		"""
		memo:
			P -> global points
			Pc -> points in camera cs
			s -> Pc[2]
		solve:
			tP = R * tPc + T
			tPc = inv_R * (tP - T)
			Pc = (P - T) * R
			s * tU = M * tPc
			U = Pc * tM / s
		:param P: (n, 3) array like, in global coordsys
		:param z_near:
		:param z_far:
		:param force_in_frame: ignore frame roi
		:return:
			flag: true if any points left
			U: (n, 2) array in image cs, warn, zero in center
			idxes: where projected points belong to
		"""
		P = np.atleast_2d(P)
		idxes = np.arange(len(P))
		Pc = np.dot(P - self.T, self.R)

		# select points in distance range
		valid = np.logical_and(Pc[:, 2] > z_near, Pc[:, 2] < z_far)
		if not np.any(valid): return False, None, None
		idxes = idxes[valid]
		Pc = Pc[valid]

		# project points into frame
		u = Pc[:, 0] / Pc[:, 2] * self.fx
		v = Pc[:, 1] / Pc[:, 2] * self.fy
		U = np.asarray([u, v], P.dtype).T
		if not force_in_frame: return True, U, idxes

		# select points in frame range
		valid_u = np.logical_and(U[:, 0] > self.roi[0], U[:, 0] < self.roi[2])
		valid_v = np.logical_and(U[:, 1] > self.roi[1], U[:, 1] < self.roi[3])
		valid = np.logical_and(valid_u, valid_v)
		if not np.any(valid): return False, None, None
		return True, U[valid], idxes[valid]

	def reconstruction(self, U, z):
		"""
		solve:
			1) U -> Pc
				x = u / fx * z
				y = v / fy * z
			2) P = Pc * tR + T
		:param U: (n, 2) array, in image coordSys, warn, zero in center
		:param z: (n, ) array, axis-z in camera cs
		:return:
			P: (n, 3) array, in global coordSys
		"""
		U = np.atleast_2d(U)
		total_P = len(U)
		total_z = len(z)
		assert np.min(z) > NUM_ZERO, log.error("positive zs required .. ")
		assert total_z == total_P, log.error(f"count pts({total_P}) != z({total_z}) ..")

		x = U[:, 0] / self.fx * z
		y = U[:, 1] / self.fy * z
		Pc = np.asarray([x, y, z], U.dtype).T
		return Pc.dot(self.tR) + self.T


if __name__ == '__main__':
	# folder_o = "/home/kai/Desktop/DailySource/M23_04/D2304_02/out/"
	# canvas = SpaceCanvas(folder_o)

	# def perspective_mat(w, h):
	# 	src = np.asarray([
	# 		[0, 0],
	# 		[0, h],
	# 		[w, h],
	# 		[w, 0]
	# 	], np.float64)
	# 	a, b = 16, 9
	# 	dst = np.asarray([
	# 		[-a, -b],
	# 		[-a, b],
	# 		[a, b],
	# 		[a, -b]
	# 	], np.float64)
	# 	H = getPerspectiveMat(src, dst)
	# 	return H

	def main():
		# total = 4000
		# w, h = 1600, 900
		# pts = np.random.random((total, 3)) * (100, 100, 100) - (50, 50, 0)
		# # c2w = np.asmatrix("0 -1 0 0; 0 0 -1 0; 1 0 0 0; 0 0 0 1")
		# # canvas.add_wireCamera("camera", mat = c2w)
		# valid = np.full(total, True)
		#
		# cam = CVCamera((1000, 1000, 800, 450), (w, h))
		# flag, pis, idxes = cam.projection(pts)
		# assert flag
		#
		# valid[idxes] = False
		# canvas.add_point("points_out", pts[valid], color = KaisColor.axis(0))
		# canvas.add_point("points_in", pts[idxes], color = KaisColor.axis(1))

		# mat = perspective_mat(w, h)
		# pis_dst = cam.perspective(mat, pis)
		# pcs_dst = np.full((len(idxes), 3), -10, np.float64)
		# pcs_dst[:, :2] = pis_dst
		# pis_dst *= 3
		# canvas.add_point("perspective", pcs_dst, color = KaisColor.axis(2))
		# canvas.add_line("lines1", pcs_dst, pts[idxes])

		# pcs_dst = cam.reconstruction(pis, 10)
		# canvas.add_point("recover", pcs_dst, color = KaisColor.axis(2))
		# canvas.add_line("lines2", pcs_dst, pts[idxes])
		pass


	main()
	# canvas.save()
	pass
