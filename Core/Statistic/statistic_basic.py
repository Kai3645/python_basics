import math

import numpy as np

from Core.Basic import errPrint, NUM_ERROR


def CCR(w):
	"""
	Cumulative Contribution Ratio
	:param w:
	:return:
	"""
	ccr = np.copy(w)
	for i in range(1, len(w)):
		ccr[i] += ccr[i - 1]
	return ccr / ccr[-1]


def PCA(X, ccr_th: float = 1):
	"""
	Principal Component Analysis
	:param X: (N, dim) data
	:param ccr_th: ccr threshold
	:return:
		ave:
		s:
		u:
	"""
	X = np.atleast_2d(X)
	N, dim = X.shape
	assert N >= dim, f">> err, data num {N} < dim {dim} .."

	ave = np.mean(X, axis = 0)
	X = X - ave
	V = np.dot(X.T, X) / (N - 1)
	u, s, _ = np.linalg.svd(V)

	if ccr_th == 1: return ave, s, u
	ccr = CCR(s)
	valid = ccr < ccr_th
	return ave, s[valid], u[:, valid]


def LPP(X, th: float = -1, ccr_th: float = 1):
	"""
	Locality Preserving Projection
	:param X: (N, dim) data
	:param th: safety reciprocal coe
	:param ccr_th: ccr threshold
	:return:
	"""
	X = np.atleast_2d(X)
	N, dim = X.shape
	assert N >= dim, f">> err, data num {N} < dim {dim} .."

	ave = np.mean(X, axis = 0)
	X = X - ave

	tmp = np.arange(N)
	idx1, idx2 = np.meshgrid(tmp, tmp)
	idx1 = idx1.reshape(-1)
	idx2 = idx2.reshape(-1)

	dX = X[idx1] - X[idx2]
	dS = np.sum(dX * dX, axis = 1)
	if th <= NUM_ERROR:
		th = max(float(np.median(dS)) * 2, 1) * 3
	A = np.exp(dS / -th)
	A = A.reshape((N, N))
	D = np.diag(np.sum(A, axis = 1))
	L = D - A
	T1 = np.dot(np.dot(X.T, D), X)
	T2 = np.dot(np.dot(X.T, L), X)
	V = np.dot(np.linalg.inv(T1), T2)
	u, s, _ = np.linalg.svd(V)

	if ccr_th == 1: return ave, s, u
	ccrs = CCR(s)
	valid = ccrs < ccr_th
	return ave, s[valid], u[:, :valid]


def LinearDiscriminantAnalysis(X, classIndexes, ccr_th: float = 1):
	"""
	:param X: (N, dim) data
	:param classIndexes: (M, Ni) class indexes
	:param ccr_th: ccr threshold
	:return:
	"""
	X = np.atleast_2d(X)
	N, dim = X.shape

	if N < dim:
		errPrint(f">> err, data num {N} < dim {dim} ..")
		return None, None, None

	ave = np.mean(X, axis = 0)
	X = np.asmatrix(X - ave)

	Sw = np.asmatrix(np.zeros((dim, dim)))
	Sb = np.asmatrix(np.zeros((dim, dim)))
	for idxes in classIndexes:
		Xi = X[idxes]
		Ni, _ = Xi.shape
		ave_i = np.mean(Xi, axis = 0)
		Xi = Xi - ave_i
		Sw += Xi.transpose() * Xi
		delta_ave = ave_i - ave
		Sb += delta_ave * delta_ave.transpose() * Ni
	Vw = Sw / N
	Vb = Sb / N
	V = np.linalg.inv(Vw) * Vb
	u, s, _ = np.linalg.svd(V)

	if ccr_th == 1: return ave, s, u
	ccrs = CCR(s)
	valid = ccrs < ccr_th
	return ave, s[valid], u[:, valid]


LDA = FDA = LinearDiscriminantAnalysis


def merge_covariance(x: np.ndarray, P: np.matrix, y: np.ndarray, Q: np.matrix):
	S = P + Q
	K = P * np.linalg.inv(S)
	Dx = K * np.asmatrix(y - x).transpose()
	z = x + np.asarray(Dx).transpose().ravel()
	R = P - K * S * K.transpose()
	return z, R


def histogram(x, *, normalize: bool = True):
	"""
	IQR --> https://en.wikipedia.org/wiki/Interquartile_range
	FD --> https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule
	SR --> https://www.vosesoftware.com/riskwiki/Sturgesrule.php
	:param x:
	:param normalize:
	:return:
	"""
	n = len(x)
	idxes = np.argsort(x)
	i0 = idxes[0]
	i1 = idxes[round(n / 4)]
	i3 = idxes[min(round(n * 3 / 4), n - 1)]
	i4 = idxes[-1]

	IQR = x[i3] - x[i1]
	FD = 2 * IQR / pow(n, 1 / 3)
	R = x[i4] - x[i0]
	SR = R / (1 + 3.322 * math.log10(n))
	bin_width = min(SR, FD)

	lim_L = int(x[i0] / bin_width) - 2
	lim_R = int(x[i4] / bin_width) + 2
	length = lim_R - lim_L

	tmp_data = x - lim_L * bin_width
	vote_L = ((tmp_data + NUM_ERROR) / bin_width).astype(int)
	vote_R = ((tmp_data - NUM_ERROR) / bin_width).astype(int)
	vote = np.concatenate((vote_L, vote_R))
	index, count = np.unique(vote, return_counts = True)

	x_dst = (np.arange(length) + lim_L + 0.5) * bin_width
	y_dst = np.zeros(length)
	y_dst[index] = count  # todo: IndexError: index 10872728 is out of bounds for axis 0 with size 10872725
	if normalize: y_dst /= n * 2 * bin_width
	return x_dst, y_dst


def fit_normal_distribution(X, conf_id: int = 4):
	"""
	:param X: (n, ) array like
	:param conf_id: confidence id, {1, 2, 3, 4}
	:return:
	"""
	n = len(X)
	assert n > 1
	Qis = [int(max(1, n // 4)), int(max(1, n // 2)), int(n / 4 * 3), n]
	Qi = Qis[conf_id - 1]
	X = np.asarray(X)

	best_id = n
	mean = np.mean(X)
	idxes = np.argsort(np.abs(X - mean))
	for i in range(20):
		if idxes[0] == best_id: break
		best_id = idxes[0]
		mean = np.mean(X[idxes[:Qi]])
		idxes = np.argsort(np.abs(X - mean))
	delta = X[idxes[:Qi]] - mean
	sigma = math.sqrt(delta.dot(delta) / len(delta))
	return mean, sigma, best_id
