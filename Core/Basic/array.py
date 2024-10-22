import string
from typing import Literal

import numpy as np

from Core.Basic.log import KaisLog
from Core.Basic.scalars import NUM_ERROR

log = KaisLog.get_log()


def num_to_bytes(arr, *, decimal: int = 3, length = 4, byteorder: Literal["little", "big"] = "little"):
	scale = 10 ** decimal
	arr = np.asanyarray(arr)
	if arr.ndim == 0:
		num = round(arr * scale)
		return num.to_bytes(length, byteorder, signed = True)
	arr = (arr * scale).round()
	tmp = [int(a).to_bytes(length, byteorder, signed = True) for a in arr]
	return bytes().join(tmp)


def bytes_to_num(src, *, decimal: int = 3, length = 4, byteorder: Literal["little", "big"] = "little"):
	scale = 10 ** decimal
	src = [int.from_bytes(src[i: i + length], byteorder = byteorder, signed = True) for i in
	       range(0, len(src), length)]
	arr = np.asanyarray(src) / scale
	if len(arr) == 1: return arr[0]
	return arr


def num_to_str(arr, *, decimal: int = 3, sep: str = ", "):
	f = f"%.{str(decimal)}f"  # LOL :D
	arr = np.asanyarray(arr)
	if arr.ndim == 0: return f % arr
	return sep.join([f % a for a in arr])


def str_to_num(src: str, *, decimal: int = 3, sep: str = ","):
	if decimal < 1: decimal = None
	arr = np.asanyarray(src.split(sep), dtype = float).round(decimal)
	if len(arr) == 1: return arr[0]
	return arr


def arg_interpolate(res, src, isClosed: bool = False):
	"""
	interpolate x-array into base-array and return nearest indexes
	res = [0, 1, 3, 4]
	src = 2
	-> idx_L = 1
	-> idx_R = 2
	:param res: sorted independent 1d_array
	:param src: required 1d_array
	:param isClosed: True for out of range not allowed
	:return:
		idxes_l: left side indexes, -1 for out of range
		idxes_r: right side indexes, -1 for out of range
	"""
	res = np.asanyarray(res)
	src = np.asanyarray(src)
	
	assert res.ndim > 0, log.error("@arg_interpolate: res, 1d like  array required")
	
	len_res = len(res)
	
	assert len_res > 1, log.error("@arg_interpolate: res, at least 2 members")
	
	if src.ndim == 0:
		idx_L, idx_R = -1, -1
		for i in range(len_res):
			if src > res[i]: continue
			idx_L = i - 1
			idx_R = i
			break
		if idx_R == 0:
			if isClosed: return -1, -1
			else: return 0, 1
		if idx_R == -1:
			if isClosed: return -1, -1
			else: return len_res - 2, len_res - 1
		return idx_L, idx_R
	
	len_src = len(src)
	
	idx_L = np.full(len_src, -1, int)
	idx_R = np.full(len_src, -1, int)
	
	work_array = np.concatenate((res, src))
	print(work_array)
	idx_save = np.concatenate((np.full(len_res, -1, int), np.arange(len_src, dtype = int)))
	print(idx_save)
	idx_sorted = idx_save[work_array.argsort()]
	print(idx_sorted)
	
	i_res = 0
	for i_src in idx_sorted:
		if i_src < 0:
			i_res += 1
			continue
		idx_L[i_src] = i_res - 1
		idx_R[i_src] = i_res
	# where idx_R == 0, over left edge
	valid = idx_R == 0
	if isClosed:
		idx_L[valid] = -1
		idx_R[valid] = -1
	else:
		idx_L[valid] = 0
		idx_R[valid] = 1
	# where idx_R == len_res, over right edge
	valid = idx_R == len_res
	if isClosed:
		idx_L[valid] = -1
		idx_R[valid] = -1
	else:
		idx_L[valid] = len_res - 2
		idx_R[valid] = len_res - 1
	return idx_L, idx_R


def scaling(arr_L, arr_R, src):
	"""
	basic: t = (A - L) / (R - L)
	:param arr_L: left num
	:param arr_R: right num
	:param src:
	:return: t:
	"""
	arr_L = np.asanyarray(arr_L)
	arr_R = np.asanyarray(arr_R)
	src = np.asanyarray(src)
	assert len(src) == len(arr_L) == len(arr_R), log.error("@scaling: len(x1) = len(x1) = len(x1)")
	
	den = arr_R - arr_L
	assert np.sum(den <= 0), log.error("@scaling: math err, den = 0")
	
	return (src - arr_L) / den


def valid_intercept(arr, min_value, max_value):
	"""
	:param arr:
	:param min_value:
	:param max_value:
	:return indexes:
	"""
	valid1 = arr > min_value - NUM_ERROR
	valid2 = arr < max_value + NUM_ERROR
	return np.logical_and(valid1, valid2)


def arg_present(arr, request: tuple = (3, 3, 3)):
	"""
	when u want to display a very long data-array,
	this func picks sample index at [start, body, end]
	example: present number 0 - 100
	print like 0, 1, 2, ..,49, 50, 51, .., 98, 99, 100
	:param arr:
	:param request: of rows to display
	:return
		start:
		body:
		enc:
	"""
	length = len(arr)
	idxes = np.arange(length)
	
	mid_L = request[0]
	mid_R = mid_L + request[1]
	end_L = mid_R
	end_R = end_L + request[2]
	if mid_L >= length:
		return idxes, idxes[-1], idxes[-1]
	if mid_R >= length:
		return idxes[:request[0]], idxes[request[0]:], idxes[-1]
	if end_R >= length:
		return idxes[:request[0]], idxes[mid_L:mid_R], idxes[end_L:]
	mid_L = request[0] + (length - end_R) // 2
	mid_R = mid_L + request[1]
	end_L = length - request[2]
	return idxes[:request[0]], idxes[mid_L:mid_R], idxes[end_L:]


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


def passwd_generator(length, level: int = 4):
	"""
	password generator
	:param length: of password, at least 6
	:param level:
		0 -> digits (12, ) / 12
		1 -> lower letters (12, ) / 12
		2 -> lower letters + digits (x, 6) / 12
		3 -> lower + upper letters + digits (x, 5, 3) / 12
		4 -> lower + upper letters + digits + punctuation (x, 5, 3, 2) / 12
	:return:
	"""
	assert length >= 6, log.error("@passwd_generator: length must be >= 6")
	
	request = [0, 0, 0, 0]  # [1, a, A, *]
	if level == 0: request[0] = length
	elif level == 1: request[1] = length
	elif level == 2:
		request[1] = int(length / 2)
		request[0] = length - request[1]
	elif level == 3:
		request[2] = int(length / 4 - 0.33)
		request[1] = int(length / 2.2 + 0.83)
		request[0] = length - request[1] - request[2]
	else:
		request[3] = int(length / 6)
		request[2] = int(length / 4 - 0.33)
		request[1] = int(length / 2.2 + 0.83)
		request[0] = length - request[1] - request[2] - request[3]
	
	consonant = list("bcdfghjkmnpqrstvwxyz")
	consonant_ = list("BCDFGHJKLMNPQRSTVWXYZ")
	vowel = list("aeiou")
	vowel_ = list("AEIU")
	digits = list("0123456789")
	punctuation = list("!#&()*+<=>?[]^{|}")
	
	pws = []
	if request[0] > 0:  # digits
		w = np.random.choice(digits, request[0], True)
		pws.append("".join(w))
	
	k1 = round(request[1] * 0.6)
	if k1 > 0:  # lower letters (consonant)
		len_ = int(len(consonant) * 0.8)
		if k1 > len_:
			w = np.random.choice(consonant, len_, False)
			pws.append("".join(w))
			w = np.random.choice(consonant, k1 - len_, True)
			pws.append("".join(w))
		else:
			w = np.random.choice(consonant, k1, False)
			pws.append("".join(w))
	k2 = request[1] - k1
	if k2 > 0:  # lower letters (vowel)
		len_ = int(len(vowel) * 0.8)
		if k2 > len_:
			w = np.random.choice(vowel, len_, False)
			pws.append("".join(w))
			w = np.random.choice(vowel, k2 - len_, True)
			pws.append("".join(w))
		else:
			w = np.random.choice(vowel, k2, False)
			pws.append("".join(w))
	
	k1 = round(request[2] * 0.6)
	if k1 > 0:  # upper letters (consonant)
		len_ = int(len(consonant_) * 0.8)
		if k1 > len_:
			w = np.random.choice(consonant_, len_, False)
			pws.append("".join(w))
			w = np.random.choice(consonant_, k1 - len_, True)
			pws.append("".join(w))
		else:
			w = np.random.choice(consonant_, k1, False)
			pws.append("".join(w))
	k2 = request[2] - k1
	if k2 > 0:  # upper letters (vowel)
		len_ = int(len(vowel_) * 0.8)
		if k2 > len_:
			w = np.random.choice(vowel_, len_, False)
			pws.append("".join(w))
			w = np.random.choice(vowel_, k2 - len_, True)
			pws.append("".join(w))
		else:
			w = np.random.choice(vowel_, k2, False)
			pws.append("".join(w))
	
	if request[3] > 0:
		len_ = len(punctuation)
		if request[3] > len_:
			w = np.random.choice(punctuation, len_, False)
			pws.append("".join(w))
			w = np.random.choice(punctuation, request[3] - len_, True)
			pws.append("".join(w))
		else:
			w = np.random.choice(punctuation, request[3], False)
			pws.append("".join(w))
	
	pws = list("".join(pws))
	while True:
		pws = np.random.choice(pws, length, False)
		if level < 2:
			break
		if level >= 4 and pws[-1] in punctuation:
			continue
		if pws[0] in consonant:
			break
	pw = "".join(pws)
	return pw
