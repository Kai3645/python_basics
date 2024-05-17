import numpy as np

from Core.Basic import NUM_ZERO


def normalize_universal(U):
	"""
	based on:
		@ ORB_SLAM2_detailed_comments/src/Initializer.cc
		@ void Initializer::Normalize(..
	memo 1:
		tU' = K * tU
		tU' = K * C * tP/z
		tU' = C' tP/z
		=> C' = K * C
	memo 2:
		tU2' = M' * tU1' => tU2 = M * tU1
		K2 * tU2 = M' * K1 * tU1
		tU2 = (inv_K2 * M' * K1) * tU1
		M = inv_K2 * M' * K1
	:param U: (n, 2) ndarray, image keypoints
	:return:
	"""
	# 均值
	cu = np.mean(U[:, 0])
	cv = np.mean(U[:, 1])
	u_ = U[:, 0] - cu
	v_ = U[:, 1] - cv
	# 平均偏离程度
	fu = max(np.mean(np.abs(u_)), NUM_ZERO)
	fv = max(np.mean(np.abs(v_)), NUM_ZERO)
	# 尺度归一化，使得坐标的一阶绝对矩分别为1
	# 这里所谓的一阶绝对矩其实就是随机变量到取值的中心的绝对值的平均(期望)值
	u_ /= fu
	v_ /= fv
	# 归一化矩阵，前面操作的矩阵变换
	K = np.asarray([
		[1 / fu, 0, -cu / fu],
		[0, 1 / fv, -cv / fv],
		[0, 0, 1],
	], U.dtype)
	inv_K = np.asarray([
		[fu, 0, cu],
		[0, fv, cv],
		[0, 0, 1],
	], U.dtype)
	U_ = np.asarray([u_, v_], U.dtype).T
	return U_, K, inv_K


def normalize_camera(U, fx, fy, cx, cy):
	"""
	u' = (u - cx) / fx
	v' = (v - cy) / fy
	memo:
		K = inv_C
		tU' = K * tU
		tU' = K * C * tP/z
		U' = P/z
	:param U:
	:param fx:
	:param fy:
	:param cx:
	:param cy:
	:return:
	"""
	u_ = (U[:, 0] - cx) / fx
	v_ = (U[:, 1] - cy) / fy
	return np.asarray([u_, v_], U.dtype).T
