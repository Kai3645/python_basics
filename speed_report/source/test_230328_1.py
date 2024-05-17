if __name__ == '__main__':
	import numpy as np

	from Core.CodeQuality import KaisTest


	def test_1(H, U):
		U_ = np.zeros_like(U)
		den = H[2, 0] * U[:, 0] + H[2, 1] * U[:, 1] + H[2, 2]
		U_[:, 0] = (H[0, 0] * U[:, 0] + H[0, 1] * U[:, 1] + H[0, 2]) / den
		U_[:, 1] = (H[1, 0] * U[:, 0] + H[1, 1] * U[:, 1] + H[1, 2]) / den
		return U_


	# mark: best
	def test_2(H, U):
		den = H[2, 0] * U[:, 0] + H[2, 1] * U[:, 1] + H[2, 2]
		u_ = (H[0, 0] * U[:, 0] + H[0, 1] * U[:, 1] + H[0, 2]) / den
		v_ = (H[1, 0] * U[:, 0] + H[1, 1] * U[:, 1] + H[1, 2]) / den
		return np.asarray([u_, v_], U.dtype).T


	def test_3(H, U):
		U_ = U.dot(H[:, :2].T) + H[:, 2]
		U_[:, 0] /= U_[:, 2]
		U_[:, 1] /= U_[:, 2]
		return U_[:, :2]


	def main():
		H = np.random.random((3, 3)) + 0.1
		H[2, 2] = 1
		X = np.random.random((100000, 2))
		KaisTest.func_speed_test(
			members = [
				KaisTest.TestMember(
					func = test_1,
					args = (H, X),
					name = "func 01",
				),
				KaisTest.TestMember(
					func = test_2,
					args = (H, X),
					name = "func 02",
				),
				KaisTest.TestMember(
					func = test_3,
					args = (H, X),
					name = "func 03",
				),
			],
			repeat_num = 10,
			loop_num = 5,
		)

		pass


	main()
	pass
