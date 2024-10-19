import string

import numpy as np

from Core.Basic.scalars import NUM_ERROR
from Core.Basic.f_print import errPrint


def num2int(x):
	return np.array(x, np.float64).round().astype(int)


def num2bytes(x, *, decimal: int = 3, n = 4):
	scale = 10 ** decimal
	x = np.float64(x)
	x = np.round(x * scale).astype(int)
	tmp_bytes = bytes()
	for i in x:
		tmp_bytes += int(i).to_bytes(n, byteorder = "little", signed = True)
	return tmp_bytes


def bytes2num(line, *, decimal: int = 3, n = 4):
	scale = 10 ** decimal
	count = len(line) // n
	x = np.zeros(count)
	for i in range(0, count):
		x[i] = int.from_bytes(line[i * n:i * n + n], byteorder = "little", signed = True)
	return x / scale


def num2str(x, decimal: int, *, separator: str = ", "):
	f = f"%.{str(decimal)}f"  # LOL :D
	num = np.array(x).ravel()
	tmp_str = f % num[0]
	for i in num[1:]: tmp_str += separator + f % i
	return tmp_str


def str2mat(mat_str: str, shape: tuple, *, separator: str = ","):
	row = mat_str.split(separator)
	mat = np.asarray(row, np.float64)
	mat = mat.reshape(shape)
	return np.asmatrix(mat)


def argInterpolate(base, x, *, safe_mode: bool = False):
	"""
	interpolate x-array into base-array and return nearest indexes
	:param base: sorted independent 1d_array
	:param x: required 1d_array
	:param safe_mode: True for checking edge value whether out of range
	:return:
		idxes_l: left side indexes
		idxes_r: right side indexes
	"""
	if safe_mode:
		if np.min(x) < base[0] or np.max(x) > base[-1]:
			errPrint(">> required array is out of range ..")
			return None, None

	length_base = len(base)
	length_data = len(x)

	idxes_l = np.zeros(length_data, int)
	idxes_r = np.zeros(length_data, int)

	work_array = np.concatenate((base, x))
	idxes_save = np.concatenate((np.full(len(base), -1), np.arange(length_data)))
	idxes_sort = work_array.argsort()
	flags_save = idxes_save[idxes_sort]

	count = 0
	for i, j in enumerate(flags_save):
		if j < 0:
			count += 1
			continue
		idxes_l[j] = count - 1
		idxes_r[j] = count
	valid = idxes_l < 0
	idxes_l[valid] = 0
	idxes_r[valid] = 1
	valid = idxes_r >= length_base
	idxes_l[valid] = length_base - 2
	idxes_r[valid] = length_base - 1
	return idxes_l, idxes_r


def scaling(x1, x2, xi, *, safe_mode: bool = True):
	"""
	basic: t = (xi - x1) / (x2 - x1)
	:param x1: left num
	:param x2: right num
	:param xi:
	:param safe_mode: True for (x1 <= xi <= x2)
	:return: t:
	"""
	xi = np.atleast_1d(xi)
	x1 = np.atleast_1d(x1)
	x2 = np.atleast_1d(x2)
	length = len(xi)

	den = x2 - x1
	if np.sum(den <= 0):
		errPrint(">> math err .. ")
		return None

	t = (xi - x1) / den
	if safe_mode:
		t[t < 0] = 0
		t[t > 1] = 1

	if length > 1: return t
	return t[0]


def argIntercept(a, limit: tuple):
	"""
	:param a:
	:param limit:
	:return indexes:
	"""
	valid1 = a > limit[0] - NUM_ERROR
	valid2 = a < limit[1] + NUM_ERROR
	return np.where(valid1 & valid2)[0]


def argSample(a, nums: tuple = (3, 3, 3)):
	"""
	when u want to display a very long data-array,
	this func picks sample index at [start, body, end]
	:param a:
	:param nums: of rows to display
	:return
		start:
		body:
		enc:
	"""
	length = len(a)
	idxes = np.arange(length)
	if nums[0] + nums[2] > length:
		return idxes[:nums[0]], idxes[nums[0]:nums[0]], idxes[nums[0]:]

	mid = length - nums[2]
	body = np.random.choice(idxes[nums[0]:mid], nums[1], replace = False)
	body.sort()
	return idxes[:nums[0]], body, idxes[mid:]


def rand_str(length: int, level: int = 3, *, replace: bool = True, shape: tuple = ()):
	"""
	password creator
	:param length: of PW
	:param level:
		0 -> number only
		1 -> ascii letters only
		2 -> number with ascii letters(small)
		3 -> number with ascii letters(+ large)
		4 -> add punctuation
	:param replace: if letters could be repeatable
	:param shape: used for multi-passwords
	:return:
	"""
	letters = (
		string.digits,
		string.ascii_lowercase,
		string.ascii_letters,
		string.punctuation,
	)
	idxes = (
		[0],
		[2],
		[0, 2],
		[0, 3],
		[0, 2, 3],
	)
	char_list = ""
	for i in idxes[level]: char_list += letters[i]
	shape_ = np.append(shape, length).astype(int)
	dst = np.random.choice(list(char_list), shape_, replace = replace)
	if len(shape) > 0:
		return dst.view((str, length)).reshape(shape)
	return dst.view((str, length))[0]


def password(requires, replace: bool = True):
	letters = (
		string.digits,
		string.ascii_lowercase,
		string.ascii_uppercase,
		string.punctuation,
	)
	requires = np.atleast_1d(requires).astype(int)
	length = sum(requires)
	pw = np.empty(0, "<U1")
	for le, re in zip(letters, requires):
		pw = np.concatenate((pw, np.random.choice(list(le), re, replace = replace)))
	rand_index = np.argsort(np.random.random(length), kind = "quicksort")
	return pw[rand_index].view((str, length))[0]
