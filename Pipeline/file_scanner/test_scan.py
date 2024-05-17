if __name__ == '__main__':
	"""
	cmd: python "/home/kai/Documents/GitHub/ACRS_MMS/Python/Pipeline/file_scanner/test_scan.py"
	argv[1]: folder for background images
	argv[2]: folder for saving images
	argv[3]: image path
	argv[4]: weight of document
	argv[5]: height of document
	"""
	import sys

	sys.path.append("/home/kai/Documents/GitHub/ACRS_MMS/Python")

	import cv2

	from Core.Basic import str2folder, listdir
	from Core.ComputerVision.Application.ScanDocument import test_scan, get_backgrounds, warp

	arg_len = len(sys.argv)
	if arg_len < 6:
		print(">> path err ..")
		exit(-1)

	folder_back = str2folder(sys.argv[1])
	folder_save = str2folder(sys.argv[2])
	path = sys.argv[3]
	W = float(sys.argv[4])
	H = float(sys.argv[5])

	imgs = []
	files = listdir(folder_back)
	for f in files: imgs.append(cv2.imread(folder_back + f))
	bgs = get_backgrounds(imgs)

	src = cv2.imread(path)
	dst, edge, corners = test_scan(src, bgs)
	cv2.imwrite(folder_save + "detected_corners.jpeg", dst)
	cv2.imwrite(folder_save + "detected_edges.jpeg", edge)

	dst = warp(src, corners, W, H, 0, 330, [0, 1, 2, 3])
	cv2.imwrite(folder_save + "warped.jpeg", dst)
