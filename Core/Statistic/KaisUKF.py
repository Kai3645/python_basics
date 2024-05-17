import math

import numpy as np
from scipy.linalg import sqrtm

from Core.Basic import num2str, errInfo


class UKFPara:
	def __init__(self, dim_n: int, *, alpha, beta, kappa):
		self.dim_n = dim_n
		self.dim_s = dim_s = 2 * dim_n + 1

		self.alpha = alpha
		self.beta = beta
		self.kappa = kappa
		self.lam = lam = alpha * alpha * (dim_n + kappa) - dim_n

		self.Wm = np.zeros(dim_s)
		self.Wm[0] = lam / (lam + dim_n)
		self.Wm[1:] = 0.5 / (lam + dim_n)

		self.Wc = np.zeros(dim_s)
		self.Wc[0] = lam / (lam + dim_n) + 1 - alpha * alpha + beta
		self.Wc[1:] = 0.5 / (lam + dim_n)

	def generate_sigma_points(self, mean, covar):
		dim = len(mean)
		tmp_root = np.asarray(sqrtm(covar)) * math.sqrt(self.lam + dim)
		if tmp_root[0, 0] == math.nan:
			print(errInfo("error in generate_sigma_points"))
			return None
		sigma = np.zeros((self.dim_s, dim))
		sigma[:, :] = mean
		for i in range(dim):
			sigma[i + 1, :] += tmp_root[i]
			sigma[i + 1 + dim, :] -= tmp_root[i]
		return sigma

	def projection_func(self, func, src_sigma, **kwargs):
		tmp = func(src_sigma[0], **kwargs)
		if tmp is None: return None, None, None
		dim = len(tmp)
		dst_sigma = np.zeros((self.dim_s, dim))
		dst_sigma[0, :] = tmp
		dst_mean = self.Wm[0] * tmp
		for i in range(1, self.dim_s):
			tmp = func(src_sigma[i], **kwargs)
			if tmp is None: return None, None, None
			dst_sigma[i, :] = tmp
			dst_mean += self.Wm[i] * dst_sigma[i]

		dst_covar = np.asmatrix(np.zeros((dim, dim)))
		for i in range(self.dim_s):
			tmp = np.asmatrix(dst_sigma[i] - dst_mean)
			dst_covar += self.Wc[i] * tmp.transpose() * tmp
		return dst_mean, dst_covar, dst_sigma


class KaisUKF:
	def __init__(self, dim_x: int, dim_u: int, dim_z: int, **kwargs):
		self.dim_x = dim_x
		self.dim_y = dim_x
		self.dim_z = dim_z
		self.dim_u = dim_u
		self.dim_a = dim_a = dim_x + dim_u

		self.para = UKFPara(
			dim_a,
			alpha = kwargs.get("alpha", 1.0),
			beta = kwargs.get("beta", 2.0),
			kappa = kwargs.get("kappa", 1.0)
		)

		self.log_x = []
		self.log_z = []

	def save_to_csv(self, folder: str):
		labels = ["z", "x"]
		data = [self.log_z, self.log_x]
		for a, d in zip(labels, data):
			path = folder + f"UKF_data_{a}.csv"
			fw = open(path, "w")
			for mean, covar in d:
				tmp = np.asarray(mean).ravel()
				fw.write((num2str(tmp, 10) + ",|,"))
				tmp = np.asarray(covar).ravel()
				fw.write((num2str(tmp, 10) + "\n"))
			fw.close()

	def special_init(self, x_mean, x_covar, func_y2z, **kwargs):
		# X = Y -> Z -> X+U = A -> Y -> Z -> X+U = A -> Y ...
		y_mean, y_covar = x_mean, x_covar
		y_sigma = self.para.generate_sigma_points(y_mean, y_covar)
		z_data = self.para.projection_func(func_y2z, y_sigma, **kwargs)

		return (y_mean, y_covar, y_sigma), z_data

	def update_sigma(self, x_mean, x_covar, u_covar, func_a2y, func_y2z, **kwargs):
		# X+U = A -> Y -> Z -> X+U = A -> Y -> Z -> X ...
		a_mean = np.zeros(self.dim_a)
		a_mean[:self.dim_x] = x_mean
		a_covar = np.asmatrix(np.zeros((self.dim_a, self.dim_a)))
		a_covar[:self.dim_x, :self.dim_x] = x_covar
		a_covar[self.dim_x:, self.dim_x:] = u_covar
		a_sigma = self.para.generate_sigma_points(a_mean, a_covar)

		y_data = self.para.projection_func(func_a2y, a_sigma, **kwargs)
		y_mean, y_covar, y_sigma = y_data
		z_data = self.para.projection_func(func_y2z, y_sigma, **kwargs)
		return y_data, z_data

	def update_matrix(self, y_data, z_data, o_mean, o_covar):
		y_mean, y_covar, y_sigma = y_data
		z_mean, z_covar, z_sigma = z_data
		T = np.asmatrix(np.zeros((self.dim_y, self.dim_z)))
		for i in range(self.para.dim_s):
			tmp_y = np.asmatrix(y_sigma[i] - y_mean).transpose()
			tmp_z = np.asmatrix(z_sigma[i] - z_mean)
			T += tmp_y * tmp_z * self.para.Wc[i]

		S = z_covar + o_covar
		K = T * np.linalg.inv(S)
		Dlt = K * np.asmatrix(o_mean - z_mean).transpose()
		x_mean = y_mean + np.asarray(Dlt).transpose().ravel()
		x_covar = y_covar - K * S * K.transpose()

		self.log_z.append((z_mean, z_covar))
		self.log_x.append((x_mean, x_covar))
		return x_mean, x_covar
