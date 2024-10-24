import math
from itertools import combinations

import numpy as np

from Core.Basic import is_zero
from Core.Math import gaussian_elimination
from Core.Statistic.statistic_basic import PCA


def fit_plane_(points):
	"""
	| X10  ..  X1i  -1 | | nv0 |
	| X20  ..  X2i  -1 | |  :  | = 0
	| X30  ..  X3i  -1 | | nvi |
	                     |  d  |
	&& exist any( nv0, .., nvi, d ) != 0
	:param points: (n, dim) array like
	:return: flag, nv, d
	"""
	N, dim = points.shape
	M = np.zeros((dim, dim + 2))
	M[:dim, :dim] = points[:dim]
	M[:dim, dim] = -1
	x, dof, _ = gaussian_elimination(M, value = 1)
	if x is None: return False, None, 0
	x /= np.linalg.norm(x[:-1])
	if x[-1] < 0: x = -x
	return True, x[:-1], x[-1]


def fit_plane_LSM(points):
	"""
	Least Square Method
	set nv[flag_i] = 1
	| X10  ..  X1i | | nv0 |   | -X1[flag_i] |
	|  :        :  | |  :  | = |    :        |
	| Xn0  ..  Xni | | nvi |   | -Xn[flag_i] |
	:param points: (n, dim) array like
	:return: flag, nv, d
	"""
	N, dim = points.shape
	ave = np.mean(points, axis = 0)
	P = points - ave
	
	weights = np.sum(P * P, axis = 0)
	order = np.arange(dim)
	flag_i = np.argmin(weights)
	order = order[order != flag_i]
	
	M = np.zeros((dim - 1, dim))
	A = P[:, order]
	tA = A.transpose()
	M[:, :-1] = np.dot(tA, A)
	M[:, -1] = np.dot(tA, -P[:, flag_i])
	x, dof, _ = gaussian_elimination(M, value = 1)
	if x is None: return False, None, 0
	nv = np.ones(dim)
	nv[order] = x
	nv /= np.linalg.norm(nv)
	d = np.dot(nv, ave)
	if d < 0: return True, -nv, -d
	return True, nv, d


def fit_plane_Random(points):
	N, dim = points.shape
	idxes = np.random.choice(np.arange(N), dim, replace = False)
	return fit_plane_(points[idxes])


def fit_plane_RANSAC(points, iter_num: int):
	# todo: can be upgraded
	N, dim = points.shape
	random_indexes = (np.random.random(N)).argsort()
	points = points[random_indexes]
	
	# solve C(num, dim) < iter_num, magic number
	lnP = math.log(iter_num) + math.log((np.arange(2, dim + 1)).prod())
	num = min(int(math.exp(lnP / dim) + (0.5 * dim) - 0.5), N)
	idxes_pool = list(combinations(range(num), dim))
	idxes_pool = (np.asarray(idxes_pool) * (N / num)).astype(int)
	
	flag = False
	best_n = None
	best_d = 0
	dis_th = 99
	Q3i = int(N / 4 * 3)
	for idxes in idxes_pool:
		flag, nv, d = fit_plane_(points[idxes])
		if not flag: continue
		D = np.abs(np.dot(points, nv) - d)
		inlier = np.sum(D < dis_th)
		D.sort()
		if inlier > Q3i:
			best_n = nv
			best_d = d
			dis_th = min(dis_th, D[Q3i])
	return flag, best_n, best_d


def fit_plane_PCA(points):
	ave, s, u = PCA(points, ccr_th = 1)
	nv = u[:, -1]
	nv /= np.linalg.norm(nv)
	d = np.dot(nv, ave)
	if d < 0: return True, -nv, -d
	return True, nv, np.dot(nv, ave)


def compensate_dth(points, fit_func, loop_th):
	N, dim = points.shape
	Q3i = int(N / 4 * 3)
	
	flag, nv, d = fit_func(points)
	D = np.abs(np.dot(points, nv) - d)
	rms2_pre = np.sum(D * D) / N
	for i in range(loop_th):
		idxes = D.argsort()
		flag, nv, d = fit_func(points[idxes[:Q3i]])
		D = np.abs(np.dot(points, nv) - d)
		rms2 = np.sum(D * D) / N
		# print("compensate", i, round(rms2_pre, 4), round(rms2, 4))
		if is_zero(rms2 - rms2_pre): return flag, nv, d
		rms2_pre = rms2
	return flag, nv, d


def fit_plane(points, fit_method: str = "none", com_method: str = "none", **kwargs):
	"""
	N-dim points -> (N-1)-dim plane
	fitting function
	solve: N * X = d -> N, d
	current best plan ==>> "pca" + "dth"
	:param points: (n, dim) array
	:param fit_method:
		'none': Least Square Method
		'fast': use top 3 points only
		'lsm': Least Square Method
		'random': use random 3 points
		'ransac': non repeating random selecting
		'pca': maybe the best
	:param com_method: ("none", "fpd") compensate method
		'none': no compensation
		'dth': using distance threshold
	:param kwargs:
	:return:
		flag: True if succeed
		v:
		d:
	"""
	points = np.atleast_2d(points)
	N, dim = points.shape
	assert N >= dim, f">> fitting err, need more {dim}D points .. "
	
	fit_method = fit_method.lower()
	fit_func_dict = {
		"none": fit_plane_LSM,
		"fast": fit_plane_,
		"lsm": fit_plane_LSM,
		"random": fit_plane_Random,
		"ransac": fit_plane_RANSAC,
		"pca": fit_plane_PCA,
	}
	assert fit_method in fit_func_dict.keys(), errInfo(">> unknown fitting method .. ")
	fit_func = fit_func_dict.get(fit_method, fit_plane_)
	if fit_method == "ransac":
		iterate_num = kwargs.get("iterate_num", 2000)
		return fit_func(points, iterate_num)
	com_method = com_method.lower()
	if com_method == "none" or fit_method not in [
		"lsm", "pca",
	]: return fit_func(points)
	
	if com_method == "dth":
		loop_th = kwargs.get("loop_th", 20)
		return compensate_dth(points, fit_func, loop_th)
	
	errPrint("err, unknown compensate method .. ")
	return False, None, 0


if __name__ == '__main__':
	from scipy.stats import multivariate_normal
	from Core.Geometry import SpacePlane
	from Core.Visualization import KaisCanvas
	from Core.Visualization import KaisColor
	from Core.Visualization import SpaceCanvas
	
	folder = "/home/kai/PycharmProjects/pyCenter/d_2022_0730/out/"
	
	
	def line_points(num, length, scale = 0.1):
		p0 = (-3, -1)
		angle = np.deg2rad(np.random.randint(-5, 45))
		v = (round(math.cos(angle), 4), round(math.sin(angle), 4))
		d = p0[1] * v[0] - p0[0] * v[1]
		print("true, nv =", [-v[1], v[0]], ", d =", round(abs(d), 4))
		err = np.random.normal(p0, scale, (num, 2))
		t = np.linspace(0, length, num)
		pts = np.tile(t, (2, 1)).T * v + err
		np.random.shuffle(pts)
		return pts
	
	
	def covar_points(num):
		a = np.deg2rad(30)
		c = math.cos(a)
		s = math.sin(a)
		mat = np.asmatrix([[c, -s], [s, c]])
		covar = mat * np.diag((6, 1)) * mat.transpose()
		mean = (3, 4)
		return multivariate_normal(mean, covar).rvs(num), mean, covar
	
	
	def line_sample(nv, d, pts):
		D = np.dot(pts, nv) - d
		K = np.tile(D, (2, 1))
		pts = pts - K.T * nv
		
		v = np.asarray([nv[1], -nv[0]])
		if len(pts) == 1: return pts[0] - v, pts[0] + v
		idxes = pts.argsort(axis = 0)[:, 0]
		extend = 0.1
		t = extend + 1
		p1 = pts[idxes[0]] * t - pts[idxes[-1]] * extend
		p2 = pts[idxes[-1]] * t - pts[idxes[0]] * extend
		return p1, p2, pts
	
	
	def main2d():
		canvas = KaisCanvas(line_width = 1.2)
		
		# >>---------- basic axis
		canvas.draw_line((-1, 0), (1, 0), color = KaisColor.axis_cnames[0])
		canvas.draw_line((0, -1), (0, 1), color = KaisColor.axis_cnames[1])
		
		# >>---------- sample points
		# points = np.random.random((10, 2))  # totally random
		# points, m, c = covar_points(1000)  # points in normal cloud
		# canvas.draw_covariance(m, c, 3)
		points = line_points(100, 20, 0.2)  # points in line
		points[-1, :] = (20, -50)  # add error data
		# points[-2, :] = (20, -15)  # add error data
		# points[-3, :] = (-10, 10)  # add error data
		canvas.draw_points(points, marker = ".", s = 10, color = "yellow", alpha = 0.6)
		
		#  >>---------- fitting methods
		#  >>---------- >>---------- method: "fast"
		# flag, nv, d = fit_plane(points, fit_method = "fast", com_method = "none")
		# print("fast, nv =", nv.round(4), ", d =", round(d, 4))
		# p1, p2, p_proj = line_sample(nv, d, points)
		# canvas.draw_line(p1, p2, label = "fast")
		
		#  >>---------- >>---------- method: "lsm"
		# flag, nv, d = fit_plane(points, fit_method = "lsm", com_method = "none")
		# print("lsm, nv =", nv.round(4), ", d =", round(d, 4))
		# p1, p2, p_proj = line_sample(nv, d, points)
		# canvas.draw_line(p1, p2, label = "lsm")
		flag, nv, d = fit_plane(points, fit_method = "lsm", com_method = "dth")
		print("lsm, nv =", nv.round(4), ", d =", round(d, 4))
		p1, p2, p_proj = line_sample(nv, d, points)
		canvas.draw_line(p1, p2, label = "lsm + dth")
		
		#  >>---------- >>---------- method: "random"
		# flag, nv, d = fit_plane(points, fit_method = "random", com_method = "none")
		# print("random, nv =", nv.round(4), ", d =", round(d, 4))
		# p1, p2, p_proj = line_sample(nv, d, points)
		# canvas.draw_line(p1, p2, label = "random")
		
		#  >>---------- >>---------- method: "ransac"
		flag, nv, d = fit_plane(points, fit_method = "ransac", com_method = "none")
		print("ransac, nv =", nv.round(4), ", d =", round(d, 4))
		p1, p2, p_proj = line_sample(nv, d, points)
		canvas.draw_line(p1, p2, label = "ransac")
		
		#  >>---------- >>---------- method: "pca"
		# flag, nv, d = fit_plane(points, fit_method = "pca", com_method = "none")
		# print("pca, nv =", nv.round(4), ", d =", round(d, 4))
		# p1, p2, p_proj = line_sample(nv, d, points)
		# canvas.draw_line(p1, p2, label = "pca")
		flag, nv, d = fit_plane(points, fit_method = "pca", com_method = "dth")
		print("pca, nv =", nv.round(4), ", d =", round(d, 4))
		p1, p2, p_proj = line_sample(nv, d, points)
		canvas.draw_line(p1, p2, label = "pca + dth")
		
		canvas.set_axis(legend_on = True, grid_on = False, legend_loc = "best")
		canvas.save(folder + "fitting.pdf")
		# canvas.show()
		canvas.close()
		pass
	
	
	def plane_points(num, width, scale = 0.5):
		nv = np.random.random(3) - 0.5
		d = round(np.random.random() * 4, 4)
		sPlane = SpacePlane.init_nd(nv, d)
		print("true, " + str(sPlane))
		p0, wx, wy = sPlane.get_pv2()
		
		U = np.random.random((num, 2)) * width - width / 2
		k = np.random.normal(0, scale, num)
		K = np.tile(k, (3, 1)).transpose()
		pts = np.dot(U, [wx, wy]) + K * sPlane.nv + p0
		np.random.shuffle(pts)
		return pts
	
	
	def main3d():
		canvas = SpaceCanvas(folder)
		
		# >>---------- basic axis
		canvas.add_axis("axis", r = 0.08, scales = (5, 5, 5))
		
		# >>---------- sample points
		total = 100
		# points = np.random.random((10, 2))  # totally random
		# points, m, c = covar_points(1000)  # points in normal cloud
		# canvas.draw_covariance(m, c, 3)
		points = plane_points(total, 20, 0.2)  # points in plane
		# >>---------- manual error point
		total -= 1
		points[total, :] = (40, 40, 120)  # add error data
		# total -= 1
		# points[total, :] = (20, -15)  # add error data
		# total -= 1
		# points[total, :] = (-10, 10)  # add error data
		canvas.add_point("points", points)
		valid_pts = np.arange(total)
		
		#  >>---------- fitting methods
		#  >>---------- >>---------- method: "fast"
		# flag, nv, d = fit_plane(points, fit_method = "fast", com_method = "dth")
		# if flag:
		# 	sPlane = SpacePlane(nv, d)
		# 	print("fast, " + str(sPlane))
		# 	p1, p2 = sPlane.adapt_sample(points[valid_pts])
		# 	canvas.add_line("fitting_fast", p1, p2)
		
		#  >>---------- >>---------- method: "lsm"
		# flag, nv, d = fit_plane(points, fit_method = "lsm", com_method = "none")
		# if flag:
		# 	sPlane = SpacePlane(nv, d)
		# 	print("lsm, " + str(sPlane))
		# 	p1, p2 = sPlane.adapt_sample(points[valid_pts])
		# 	canvas.add_line("fitting_lsm", p1, p2)
		# flag, nv, d = fit_plane(points, fit_method = "lsm", com_method = "dth")
		# if flag:
		# 	sPlane = SpacePlane(nv, d)
		# 	print("lsm, " + str(sPlane))
		# 	p1, p2 = sPlane.adapt_sample(points[valid_pts])
		# 	canvas.add_line("fitting_lsm_dth", p1, p2)
		
		#  >>---------- >>---------- method: "random"
		# flag, nv, d = fit_plane(points, fit_method = "random", com_method = "none")
		# if flag:
		# 	sPlane = SpacePlane(nv, d)
		# 	print("random, " + str(sPlane))
		# 	p1, p2 = sPlane.adapt_sample(points[valid_pts])
		# 	canvas.add_line("fitting_random", p1, p2)
		
		#  >>---------- >>---------- method: "ransac"
		# flag, nv, d = fit_plane(points, fit_method = "ransac", com_method = "none")
		# if flag:
		# 	sPlane = SpacePlane(nv, d)
		# 	print("ransac, " + str(sPlane))
		# 	p1, p2 = sPlane.adapt_sample(points[valid_pts])
		# 	canvas.add_line("fitting_ransac", p1, p2)
		
		#  >>---------- >>---------- method: "pca"
		# flag, nv, d = fit_plane(points, fit_method = "pca", com_method = "none")
		# if flag:
		# 	sPlane = SpacePlane(nv, d)
		# 	print("pca, " + str(sPlane))
		# 	p1, p2 = sPlane.adapt_sample(points[valid_pts])
		# 	canvas.add_line("fitting_pca", p1, p2)
		flag, nv, d = fit_plane(points, fit_method = "pca", com_method = "dth")
		if flag:
			sPlane = SpacePlane(nv, d)
			print("pca, " + str(sPlane))
			p1, p2 = sPlane.adapt_sample(points[valid_pts])
			canvas.add_line("fitting_pca_dth", p1, p2)
		
		canvas.save()
		pass
	
	
	# main2d()
	main3d()
	pass
