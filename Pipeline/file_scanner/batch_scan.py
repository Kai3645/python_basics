if __name__ == '__main__':
	"""
	cmd: python "/home/kai/Documents/GitHub/ACRS_MMS/Python/Pipeline/file_scanner/batch_scan.py"
	argv[1]: folder for background images
	argv[2]: folder for working images
	argv[3]: folder for saving images
	argv[4]: working image extension name
	argv[5]: saving image extension name
	argv[6]: weight of document
	argv[7]: height of document
	argv[8]: error of document edge
	argv[9]: output ppi
	argv[10]: order of corner
	"""
	import sys

	sys.path.append("/home/kai/Documents/GitHub/ACRS_MMS/Python")

	import cv2
	import numpy as np

	from Core.Basic import listdir, str2folder, multithreading
	from Core.ComputerVision.Application.ScanDocument import scanDoc, get_backgrounds

	arg_len = len(sys.argv)
	if arg_len < 11:
		print(">> input err ..")
		exit(-1)
	folder_back = str2folder(sys.argv[1])
	folder_work = str2folder(sys.argv[2])
	folder_save = str2folder(sys.argv[3])
	ext_work = sys.argv[4]
	ext_save = sys.argv[5]
	W = float(sys.argv[6])
	H = float(sys.argv[7])
	E = float(sys.argv[8])
	ppi = float(sys.argv[9])
	order = np.asarray(sys.argv[10].split(","), int)

	imgs = []
	files = listdir(folder_back, pattern = "*" + ext_work)
	for f in files: imgs.append(cv2.imread(folder_back + f))
	bgs = get_backgrounds(imgs)

	files = listdir(folder_work, pattern = "*." + ext_work)
	task_list = []
	for f in files: task_list.append((f,))


	def main_func(_f):
		_src = cv2.imread(folder_work + _f)
		_dst, _edge = scanDoc(_src, bgs, W, H, E, ppi, order)
		return _f, _src, _dst, _edge


	def result_func(_f, _src, _dst, _edge):
		_fn = _f.split(".")[0] + "." + ext_save
		if _dst is None: cv2.imwrite(folder_save + _f, _src)
		else: cv2.imwrite(folder_save + _fn, _dst)
		if _edge is not None: cv2.imwrite(folder_save + "edge_" + _fn, _edge)


	multithreading(task_list, main_func, result_func, title = "scanner", thread_num = 9, take_log = True)

	pass
