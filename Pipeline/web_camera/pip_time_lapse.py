from tqdm import tqdm

if __name__ == '__main__':
	import os
	import sys
	import time

	import cv2
	import numpy as np

	folder = sys.argv[1]
	cam_id = int(sys.argv[2])
	FPS_SET = float(sys.argv[3])
	FPS_OUT = float(sys.argv[4])

	sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

	from Core.Basic import mkdir, listdir
	from Core.Visualization import KaisColor

	folder = mkdir(folder, "TimeLapse")

	cap = cv2.VideoCapture(cam_id)
	if not cap.isOpened():
		print(">> err, can not open camera ..")
		exit(-1)

	loop = 1
	fps = FPS_SET
	DT_SET = 1 / FPS_SET
	sys_delay = 50
	delay = 1000 / FPS_SET - sys_delay

	mov_name = time.strftime("%Y%m%d%H%M%S.mov", time.localtime())
	out = cv2.VideoWriter(
		folder + mov_name, apiPreference = cv2.CAP_ANY,
		fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps = FPS_OUT, frameSize = (1024, 576),
	)  # 861030210

	text_max = 0
	ti = time.time()
	print(">> start ..")
	while True:
		print("\r" * text_max, end = "")
		ret, img = cap.read()
		if ret:
			img = cv2.GaussianBlur(img, (3, 3), 1.4)

			mask = np.zeros((160, 1100), np.uint8)
			info = time.strftime("%y.%m.%d %H:%M:%S", time.localtime())
			cv2.putText(mask, info, (23, 120), cv2.FONT_HERSHEY_DUPLEX, 3.5, 50, 15)
			cv2.putText(mask, info, (23, 120), cv2.FONT_HERSHEY_DUPLEX, 3.5, 240, 2)
			kernel = np.ones((5, 5), np.float32) / 25
			mask = cv2.filter2D(mask, -1, kernel)

			w, h, off1, off2 = 240, 44, 18, 13
			mask = cv2.resize(mask, (w, h), interpolation = cv2.INTER_CUBIC)
			mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
			sub_img = img[off2:h + off2, off1:w + off1].astype(int)
			ave = np.average(sub_img)
			if ave < 100:
				sub_img += mask
				sub_img[sub_img > 255] = 255
			elif ave < 160:
				valid = mask > 0
				sub_img[valid] = sub_img[valid] * 0.3 + mask[valid] * 1.2
				sub_img[sub_img > 255] = 255
			else:
				sub_img -= mask
				sub_img[sub_img < 0] = 0
			img[off2:h + off2, off1:w + off1, :] = sub_img.astype(np.uint8)

			img = cv2.resize(img, (1024, 576), interpolation = cv2.INTER_CUBIC)
			# cv2.imwrite(folder + f"T{loop:05d}.jpg", img)
			out.write(img)
			sub_img = cv2.GaussianBlur(img, (3, 3), 1.6)
			sub_img = cv2.resize(sub_img, (500, 281), interpolation = cv2.INTER_CUBIC)
			green = KaisColor.plotColor("skyblue", gbr = True)
			sub_img[2:39, 1:149, :] = img[:37, :148, :]
			sub_img[:40, 0, :] = green
			sub_img[:40, 149, :] = green
			sub_img[:2, 0:150, :] = green
			sub_img[39, 0:150, :] = green
			cv2.imshow("camera", sub_img)
		else: print(">> camera short circuit")
		if cv2.waitKey(int(round(delay))) == 27: break

		t_ = ti
		ti = time.time()
		dt = ti - t_
		delay -= 20 * (1 / fps - DT_SET)
		fps = fps * 0.4 + 0.6 / dt
		info = f"{loop:06d}, fps = {fps:.3f}, delay = {delay:.0f}"

		info_len = len(info)
		if info_len < text_max: info += " " * (text_max - info_len)
		else: text_max = info_len
		print(info, end = "")
		loop += 1
	time.sleep(1)
	print(">> finished ..")
