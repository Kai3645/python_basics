import numpy as np

from Core.Basic import is_zero, KaisLog

log = KaisLog.get_log()


def stairs_matrix(A, allow_swap = True):
	"""
	:param A:
	:param allow_swap:
	:return:
	"""

	A = np.atleast_2d(np.array(A, np.float64))
	n, dim = A.shape
	ids = list(range(n))
	i = 0
	for j in range(dim):
		k = i
		while is_zero(A[ids[k], j]):
			k += 1
			if k == n: break
		if k == n: continue
		# swap i, k
		ii = ids[k]
		ids[k] = ids[i]
		ids[i] = ii
		# stairs
		sample = A[ii, j:] / A[ii, j]
		ai = np.tile(A[:, j], (dim - j, 1)).transpose()
		A[:, j:] -= ai * sample
		A[ii, j:] = sample
		i += 1
		if i == n: break
	if not allow_swap: return A
	return A[ids]


def gaussian_elimination(M, *, value = 0, fast_mode = True):
	"""
	solve "AX = b" by gaussian elimination
	M = ( A : b )
	:param M: (n, dim + 1) array like
	:param value: default value for freedom
	:param fast_mode: True
	:return:
		X: result
		dof: degree of freedom
		W: free space vector
	"""
	M = np.atleast_2d(np.array(M, M.dtype))
	n, dim_ = M.shape
	dim = dim_ - 1
	ids = list(range(n))
	is_free = np.full(dim, True)

	i = 0
	for j in range(dim):
		k = i
		while is_zero(M[ids[k], j]):
			k += 1
			if k == n: break
		if k == n: continue
		is_free[j] = False
		# swap i, k
		ii = ids[k]
		ids[k] = ids[i]
		ids[i] = ii
		# stairs
		sample = M[ii, j:] / M[ii, j]
		ai = np.tile(M[:, j], (dim_ - j, 1)).T
		M[:, j:] -= ai * sample
		M[ii, j:] = sample
		i += 1
		if i == n: break
	log.debug("stairs matrix M = \n" + str(M[ids].round(4)))
	dof = np.sum(is_free)
	bound_l = dim - dof
	bound_r = n
	while is_zero(M[ids[bound_r - 1], dim]):
		bound_r -= 1
		if bound_r == 0: break
	if bound_l < bound_r: return None, -1, None

	free_idxes = np.where(is_free)[0]
	valid = np.logical_not(is_free)
	X = np.full(dim, value, M.dtype)
	X[valid] = M[ids[:bound_l], dim]
	if value != 0:
		for j in free_idxes:
			X[valid] -= M[ids[:bound_l], j] * value
	if fast_mode: return X, dof, None

	W = np.zeros((dof, dim), M.dtype)
	for i, j in enumerate(free_idxes):
		W[i, valid] -= M[ids[:bound_l], j]
		W[i, j] = 1
	for i in range(dof):
		W[i] /= np.linalg.norm(W[i])
		j = i + 1
		ai = np.tile(np.dot(W[j:], W[i]), (dim, 1)).T
		W[j:] -= ai * W[i]
	return X, dof, W


def linear_solve_svd(A, B):
	"""
	useless func
	same result with linear_solve_lsm, and slower
	solve, A * X = B
	svd, A = U * S * Vh
	    | s0         |
	S = |     ..     |
	    |         si |
	X = tVh * (S)^-1 * tU * B
	:param A: (n, dim) array like
	:param B: (n, ) array like
	:return:
		X: (dim, ) array like
	"""
	A = np.atleast_2d(np.array(A, np.float64))
	n, dim = A.shape
	assert n >= dim, log.error(f"linear_solve, n({n}) < dim({dim}) ..")
	u, s, vh = np.linalg.svd(A)
	X = np.dot(np.dot(B, u[:, :dim]) / s, vh)
	return X


def linear_solve_lsm(A, B):
	"""
	solve, A * X = B
	X = (tA * A)^-1 * tA * B
	:param A: (n, dim) array like
	:param B: (n, ) array like
	:return:
		X: (dim, ) array like
	"""
	A = np.atleast_2d(np.array(A, np.float64))
	n, dim = A.shape
	assert n >= dim, log.error(f"linear_solve, n({n}) < dim({dim}) ..")
	inv_AA = np.linalg.inv(np.dot(A.T, A))
	X = np.dot(np.dot(B, A), inv_AA.T)
	return X
