import numpy as np

from Core.Basic import num2str


class KaisEKF:
	def __init__(self, *, dim_x: int):
		self.dim_x = dim_x

		self.log_x = []
		self.log_y = []

	def save_to_csv(self, path_head: str):
		labels = ["x", "y"]
		data = [self.log_x, self.log_y]
		for a, d in zip(labels, data):
			path = path_head + f"EKF_data_{a}.csv"
			fw = open(path, "w")
			for row in d:
				if row[0] is None:
					fw.write("\n")
					continue
				fw.write(num2str(row[0], 10) + ",|,")
				tmp = np.asarray(row[1]).ravel()
				fw.write((num2str(tmp, 10) + "\n"))
			fw.close()

	def update_matrix(self, P, y, Q, dz, R, J_f, J_h):
		Y = np.asmatrix(y).transpose()
		dZ = np.asmatrix(dz).transpose()

		P_ = J_f * P * J_f.transpose() + Q
		S = J_h * P_ * J_h.transpose() + R
		K = P_ * J_h.transpose() * np.linalg.inv(S)
		x_new = np.asarray(Y + K * dZ).ravel()
		P_new = P_ - K * J_h * P_

		self.log_x.append((x_new, P_new))
		self.log_y.append((y, P_))
		return x_new, P_new
