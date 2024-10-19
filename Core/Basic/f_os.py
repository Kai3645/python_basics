import os
from fnmatch import fnmatch

import numpy as np

from Core.Basic.log import KaisLog

log = KaisLog.get_log()


def file_length(path: str, ):
	"""
	count file rows by counting '\n'
	:param path:
	:return:
	"""
	length = 0
	with open(path, 'rb') as fr:
		end = ""
		while True:  # cost 0.5 GB
			data = fr.read(0x20000000)
			if not data: break
			length += data.count(b'\n')
			end = data
		if end[-1] != b'\n': length += 1
	return length


def try_file(path: str, num: int = -1):
	"""
	try to display data for an unknown file
	:param path:
	:param num: rows to display
	:return:
	"""
	length = file_length(path)
	if num <= 0: limit = length
	else: limit = min(num, length)
	print("--------------------------------------------------")
	with open(path) as fr:
		for i, line in enumerate(fr):
			print(line, end = "")
			if i > limit: break
	print("--------------------------------------------------")
	return length


def mkdir(loc: str, new_fn: str):
	"""
	:param loc: location path
	:param new_fn: new folder name
	:return: folder path end with os separator
	"""
	assert os.path.exists(loc), f"mkdir: can not reach local path .. \"{loc}\""
	if loc[-1] != os.sep: loc += os.sep
	loc += new_fn + os.sep
	if os.path.exists(loc):
		log.info(f"mkdir: folder exist .. \"{new_fn}\"")
		return loc
	os.mkdir(loc)
	log.info(f"mkdir: create folder \"{new_fn}\"")
	return loc


def mkdir_force(folder: str, safe_mode: bool = True):
	"""
	create a folder tree
	:param folder:
	:param safe_mode:
	:return:
	"""
	if os.path.exists(folder):
		if safe_mode:
			assert len(os.listdir(folder)) > 0, f"folder not empty .. \"{folder}\""
		log.info(f">> mkdir: path exist .. \"{folder}\"")
		if folder[-1] == os.sep: return folder
		return folder + os.sep
	names = folder.split(os.sep)[1:]
	loc = ""
	for fn in names:
		loc += os.sep + fn
		if os.path.exists(loc):
			if safe_mode:
				assert len(os.listdir(loc)) > 0, f"folder not empty .. \"{loc}\""
			continue
		os.mkdir(loc)
	log.info(f">> mkdir: create path \"{loc}\"")
	return loc + os.sep


def filetype_match(path: str, filetype: str):
	"""
	use like,
	assert filetype_match(path, filetype)
	:param path:
	:param filetype:
	:return:
	"""
	if fnmatch(path, "*." + filetype): return True
	log.error(f"filetype {filetype} is required ..")
	return False


def listdir(local_path: str, *, pattern = None):
	"""
	:param local_path:
	:param pattern:
	:return: file_names:
	"""
	ignore_list = (
		".*",
		"_*",
	)
	file_names = np.asarray(os.listdir(local_path))
	invalid = np.full(len(file_names), False)
	for ig in ignore_list:
		sub_invalid = [fnmatch(n, ig) for n in file_names]
		invalid = np.logical_or(invalid, sub_invalid)
	file_names = file_names[np.logical_not(invalid)]
	if pattern is None:
		file_names.sort()
		return file_names
	valid = [fnmatch(n, pattern) for n in file_names]
	file_names = file_names[valid]
	file_names.sort()
	return file_names


def treedir(root: str, *, folder_only: bool = False, show_ignore: bool = True, space_char = " "):
	"""
	# todo: add ignore pattern list
	"┌", "└", "┐", "┘", "─", "│", "├", "┤", "┬", "┴", "┼",
	:param root:
	:param folder_only:
	:param show_ignore:
	:param space_char:
	:return: tree: string
	"""
	
	space_char2 = space_char + space_char
	
	s_head_0 = space_char2 + "│" + space_char2  # "  │"
	s_head_1 = space_char2 + "├─ "  # "  ├─ "
	s_head_2 = space_char2 + "└─ "  # "  └─ "
	s_head_3 = space_char * 5  # "     "
	
	def tree_dir(local_path, str_head = ""):
		tmp_str = ""
		if show_ignore: names = np.asarray(os.listdir(local_path))
		else: names = listdir(local_path)
		names.sort()
		
		valid = [os.path.isdir(local_path + "/" + n) for n in names]
		invalid = np.logical_not(valid)
		file_count = np.sum(invalid)
		has_file = file_count > 0
		
		folder_count = np.sum(valid)
		for i, n in enumerate(names[valid], 1):
			new_head = str_head
			tmp_str += str_head
			if has_file or folder_count > i:
				tmp_str += s_head_1 + n + "\n"
				new_head += s_head_0
			else:
				tmp_str += s_head_2 + n + "\n"
				new_head += s_head_3
			tmp_str += tree_dir(local_path + "/" + n, new_head)
		if not has_file or folder_only: return tmp_str
		
		for i, n in enumerate(names[invalid], 1):
			new_head = str_head
			tmp_str += str_head
			if file_count > i:
				tmp_str += s_head_1 + n + "\n"
				new_head += s_head_0
			else:
				tmp_str += s_head_2 + n + "\n"
				new_head += s_head_3
		return tmp_str + str_head + space_char2 + "\n"
	
	tree = space_char2 + f"@ \"{root}\"\n"
	tree += tree_dir(root)
	return tree


def treedir_markdown_style(root: str, *, show_ignore: bool = False, space_char = "."):
	"""
	"┌", "└", "┐", "┘", "─", "│", "├", "┤", "┬", "┴", "┼",
	:param root:
	:param show_ignore:
	:param space_char:
	:return: tree: string
	"""
	root = str2folder(root)
	
	space_char2 = space_char + space_char
	s_head_0 = space_char2 + "│" + space_char2  # "  │"
	s_head_1 = space_char2 + "├─ "  # "  ├─ "
	s_head_2 = space_char2 + "└─ "  # "  └─ "
	s_head_3 = space_char * 5  # "     "
	
	def tree_dir(sub_root, local_path, str_head = ""):
		tmp_str = ""
		if show_ignore: names = np.asarray(os.listdir(sub_root))
		else: names = listdir(sub_root)
		names.sort()
		
		valid = [os.path.isdir(sub_root + n) for n in names]
		invalid = np.logical_not(valid)
		file_count = np.sum(invalid)
		has_file = file_count > 0
		
		folder_count = np.sum(valid)
		for i, n in enumerate(names[valid], 1):
			new_head = str_head
			tmp_str += str_head
			if has_file or folder_count > i:
				tmp_str += s_head_1 + n + "\n"
				new_head += s_head_0
			else:
				tmp_str += s_head_2 + n + "\n"
				new_head += s_head_3
			tmp_str += tree_dir(sub_root + n + "/", local_path + n + "/", new_head)
		if not has_file: return tmp_str
		
		for i, n in enumerate(names[invalid], 1):
			new_head = str_head
			tmp_str += str_head
			file_path = local_path + n
			if file_count > i:
				tmp_str += s_head_1 + f"[{n}]({file_path})\n"
				new_head += s_head_0
			else:
				tmp_str += s_head_2 + f"[{n}]({file_path})\n"
				new_head += s_head_3
		return tmp_str + str_head + space_char2 + "\n"
	
	tree = space_char2 + f"@ \"{root}\"\n"
	tree += tree_dir(root, "./")
	return tree


def str2folder(path: str):
	"""
	:param path: abs folder path
	:return: path string end with os seperator
	"""
	log.warning("str2folder will be abandoned ..")
	log.warning("use \'path + os.sep\' instead ..")
	assert len(path) > 0, "empty path string .."
	if path[-1] != os.sep: return path + os.sep
	return path
