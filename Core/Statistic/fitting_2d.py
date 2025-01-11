import numpy as np


def fit_line2d_PCA(pts):
	"""
	xi * nx + yi * ny == d (d > 0)
	nx^2 + ny^2 == 1
	(xi, yi) = (nx, ny) * d + (ny, -nx) * t
	nx, ny = vy, -vx
	
	
	:param pts: at least 2 points
	:return: (nx, ny, d)
	"""
	if len(pts) < 2: raise ValueError("Not enough points")
	
	X = np.asarray(pts, float)
	ave = np.mean(X, axis = 0)
	X -= ave
	V = np.dot(X.T, X) / (len(X) - 1)
	u, s, _ = np.linalg.svd(V)
	
	v = u[:, 0].ravel()
	v /= np.linalg.norm(v)
	
	nx = float(v[1])
	ny = float(-v[0])
	d = float(nx * ave[0] + ny * ave[1])
	
	if d < 0: return -nx, -ny, -d
	return nx, ny, d
