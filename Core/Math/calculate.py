import math


def unwrap_angle(a, ref):
	"""
	wrap angle near to reference angle
	:param a: angle [rad]
	:param ref: reference angle [rad]
	:return: fixed angle [rad]
	"""
	pi2 = math.pi * 2
	diff = a - ref
	fix = int(abs(diff) / pi2 + 0.5) * pi2
	if diff > 0: return a - fix
	return a + fix


def integer_ratio_(x: float, digits: int = 3):
	MAX = 10 ** digits
	c = int(x)
	x -= c
	a, b = 0, 1
	m, n = 0, 1
	f = 0.0
	while n < MAX:
		g = 1.0 * m / n
		if (f - g) * (f + g - 2 * x) > 0:
			# abs(f - x) > abs(g - x)
			f = g
			a = m
			b = n
		if g > x: n += 1
		else: m += 1
	return int(c * b + a), int(b)


def integer_ratio(x: float, digits: int = 3):
	"""
	X => m / n
	salve
	X = a0 + 1 / (a1 + 1 / ( ... + 1 / (an + b))))
	:param x:
	:param digits:
	:return:
		ret_m:
		ret_n:
	"""
	end = 10 ** digits
	precision = 0.5 / end
	is_negative = x < 0
	if is_negative: x = -x
	if x < precision: return 0, 1

	ret_m, ret_n = round(x), 1
	A = []
	f = x
	m, n = ret_m, 1
	while n < end:
		ret_m, ret_n = m, n
		a = int(f)
		A.append(a)
		b_i = f - a
		if b_i < precision:
			# warming: may cause error
			if is_negative: return -ret_m, ret_n
			return ret_m, ret_n
		f = 1 / b_i
		m, n = int(f), 1
		for a in A[::-1]:
			den = m
			m = int(m * a + n)
			n = den
		# print(f"step x = {m} / {n}, list = {A} // {f}")
		pass
	x_save = ret_m / ret_n
	k = math.ceil((n - end + 1) / ret_n)
	m = int(m - ret_m * k)
	n = int(n - ret_n * k)
	x_temp = m / n
	if (x_save - x_temp) * (x_save + x_temp - 2 * x) > 0:
		# abs(x1 - x) > abs(x2 - x)
		# print(f"better x = {m} / {n}")
		ret_m, ret_n = m, n
	if is_negative: return -ret_m, ret_n
	return ret_m, ret_n


def str_ratio_num(x: float, digits: int = 3):
	m, n = integer_ratio(x, digits)
	if n == 1: return f"{m}"
	return f"{m}/{n}"


def RungeKutta(dxFunc, x, u, dt):
	"""
	:param dxFunc: dx = f(x, u)
	:param x:
	:param u:
	:param dt:
	:return:
	"""
	K0 = dt * dxFunc(x, u)
	K1 = dt * dxFunc(x + K0 / 2, u)
	K2 = dt * dxFunc(x + K1 / 2, u)
	K3 = dt * dxFunc(x + K2, u)
	dx = (K0 + 2 * K1 + 2 * K2 + K3) / 6
	return x + dx
