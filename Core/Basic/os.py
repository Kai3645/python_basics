import os
import re
from fnmatch import translate, fnmatchcase

from Core.Basic.log import KaisLog

# import numpy as np

log = KaisLog.get_log()


def file_length(path: str, n = 0x20000000):
	"""
	count file rows by counting '\n'
	:param path:
	:param n: Byte to read, default cost 0.5 GB
	:return:
	"""
	length = 0
	with open(path, 'rb') as fr:
		end = ""
		while True:
			data = fr.read(n)
			if not data: break
			length += data.count(b'\n')
			end = data
		if end[-1] != b'\n': length += 1
	return length


def try_file(path: str, n = 0x1000, with_byte = True):
	"""
	try to display data for an unknown file
	:param path:
	:param n: Byte to read
	:param with_byte:
	:return:
	"""
	x_0 = "⊠"
	x_1 = " ⊠ "
	escape = ["⊠"] * 127
	escape[0] = "∅"
	escape[9] = "↦"
	escape[10] = "↵"
	escape[13] = "↩"
	escape[32] = " "
	for i in range(32, 127):
		escape[i] = chr(i)
	step = 32
	if not with_byte: step *= 2
	
	length = file_length(path)
	with open(path, 'rb') as fr:
		while n > 0:
			line = fr.read(step)
			if with_byte:
				for c in line:
					print(f"{c:02x}", end = " ")
				print("  |  ", end = "")
			for c in line:
				i = int(c)
				if i > 126: print(x_0, end = "")
				else: print(escape[i], end = "")
			if with_byte:
				print()
				for c in line:
					i = int(c)
					if i > 126: print(x_1, end = "")
					else: print("", escape[i], "", end = "")
			print()
			n -= 32
	return length


def mkdir(loc: str, new_f: str):
	"""
	:param loc: local path, end by 'os.sep'
	:param new_f: new folder name
	:return: folder path end with os separator
	"""
	assert os.path.exists(loc), log.error(f"mkdir: not exist local path .. \"{loc}\"")
	if loc[-1] != os.sep: loc += os.sep
	loc += new_f + os.sep
	if os.path.exists(loc):
		log.warning(f"mkdir: path exist .. \"{loc}\"")
	else:
		os.mkdir(loc)
		log.info(f"mkdir: create folder \"{loc}\"")
	return loc


def _filetype_match(name: str, pat: str):
	"""
	use fnmatchcase(name, pat) instead
	this func is just a memo
	:param pat: file pattern
	:param name: file name
	:return:
	"""
	# pattern = translate(pat)
	# m = re.fullmatch(pattern, name)
	# return m is not None
	log.warning("do not use this func")
	return fnmatchcase(name, pat)


def listdir(loc: str, pats: str | list | None = None, *, ignore: str | list | None = None):
	"""
	:param loc: local path, end by 'os.sep'
	:param pats: file pattern or its list
	:param ignore: ignore file pattern or its list
	:return: file_names:
	"""
	assert os.path.exists(loc), log.error(f"mkdir: not exist local path .. \"{loc}\"")
	if loc[-1] != os.sep: loc += os.sep
	names = os.listdir(loc)
	length = len(names)
	if length == 0: return None
	
	valid = [True] * length if pats is None else [False] * length
	if pats is not None:
		if type(pats) is str: pats = [pats]
		for pat in pats:
			p = translate(pat)
			for i, fn in enumerate(names):
				# if file already in list, then pass
				if valid[i] is True: continue
				# if matched, included, valid is True
				valid[i] = re.fullmatch(p, fn) is not None
	
	if ignore is not None:
		if type(ignore) is str: ignore = [ignore]
		for pat in ignore:
			p = translate(pat)
			for i, fn in enumerate(names):
				# if file already ignored, then pass
				if valid[i] is False: continue
				# if matched, exclude, valid is False
				valid[i] = re.fullmatch(p, fn) is None
	
	rets = []
	for flag, fn in zip(valid, names):
		if flag: rets.append(fn)
	if len(rets) == 0: return None
	rets.sort()
	return rets


def treedir(loc: str, *, ignore: str | list | None = None):
	"""
	"┌", "└", "┐", "┘", "─", "│", "├", "┤", "┬", "┴", "┼",
	:param loc: local path, end by 'os.sep'
	:param ignore: ignore file pattern or its list
	:return: tree: string
	"""
	head_0 = "│   "
	head_1 = "├─ "
	head_2 = "└─ "
	head_3 = "    "
	
	def _treedir(_loc, _head = ""):
		_tree = ""
		names = listdir(_loc, ignore)
		
		folders = []
		files = []
		for n in names:
			if os.path.isdir(_loc + n):
				folders.append(n)
			else:
				files.append(n)
		count_folders = len(folders)
		count_files = len(files)
		
		for i, n in enumerate(folders, 1):
			_tree += _head
			new_head = _head
			if count_files > 0 or count_folders > i:
				_tree += head_1 + n + "\n"
				new_head += head_0
			else:
				_tree += head_2 + n + "\n"
				new_head += head_3
			_tree += _treedir(_loc + n + os.sep, new_head)
		
		for i, n in enumerate(files, 1):
			_tree += _head
			if count_files > i:
				_tree += head_1 + n + "\n"
			else:
				_tree += head_2 + n + "\n"
		return _tree
	
	assert os.path.exists(loc), log.error(f"mkdir: not exist local path .. \"{loc}\"")
	if loc[-1] != os.sep: loc += os.sep
	tree = f"@\"{loc}\"\n"
	tree += _treedir(loc)
	return tree
