import cv2
import numpy as np
from tqdm import tqdm

from Core.Basic import str2folder, KaisLog

log = KaisLog.get_log()


def video2image(video_path: str, folder_out: str, start_time: float = 0, image_type: str = "jpg"):
	folder_out = str2folder(folder_out)

	cap = cv2.VideoCapture(video_path)
	assert cap.isOpened(), f"can not open video @\"{video_path}\""

	total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	times, paths = [], []
	for _ in tqdm(range(total), desc = ">> scanning .. "):
		flag, image = cap.read()
		if not flag: break
		times.append(round(start_time + cap.get(cv2.CAP_PROP_POS_MSEC) / 1000, 4))
		paths.append(folder_out + f"{times[-1]:.4f}." + image_type)
		cv2.imwrite(paths[-1], image)

	return np.asarray(times), np.asarray(paths)


def images2video(video_path: str, image_paths, fps = 30, codec = "mp4v"):
	length = len(image_paths)
	assert length > 2

	img = cv2.imread(image_paths[-1])
	h, w = img.shape[:2]

	out = cv2.VideoWriter(
		video_path, fourcc = cv2.VideoWriter_fourcc(*codec),
		fps = fps, frameSize = (w, h),
	)  # 861030210
	for i, path in enumerate(image_paths):
		log.info(i)
		out.write(cv2.imread(path))
	out.release()


def imread_raw(path, w, h, c = 3, dtype = "uint8"):
	raw = np.fromfile(path, dtype = dtype)
	if c > 1: img = raw.reshape(h, w, c)
	else: img = raw.reshape(h, w)
	return img
