import queue
import threading
import time

import numpy as np


def multithreading(task_list, main_func, result_func, *, title = "multi", thread_num: int = 10, take_log = False):
	"""
	:param take_log:
	:param task_list:
		((a1, a2, a3, ... ), ... )
		!! need to be a tuple-like array
	:param main_func:
		func(a1, a2, a3, ... ) --> b1, b2, b3, ...
		function in threading.Lock() unlock
	:param result_func:
		func(b1, b2, b3, ...) --> None
		function in threading.Lock() locked
	:param title: default = None
		just for print
	:param thread_num: default = 10
	:return: None
	"""
	total_jobs = len(task_list)
	CD_step = max(total_jobs // 12, 1)
	jobs = queue.Queue(total_jobs)
	for w in task_list: jobs.put(w, block = True)
	thread_lock = threading.Lock()

	def push_info(info):
		tmp_str = title + " | "
		tmp_str += time.strftime("%H:%M:%S ", time.localtime())
		tmp_str += threading.current_thread().getName() + " >> " + info
		thread_lock.acquire()
		print(tmp_str)
		thread_lock.release()

	def manager():
		if jobs.qsize() == 0: return None
		task = jobs.get(block = True)
		jobs.task_done()
		num = jobs.qsize()
		if num % CD_step == 0 and take_log:
			rate = num / total_jobs * 100
			push_info(f"------------ {rate:.2f} % ------------")
		return task

	class Staff(threading.Thread):
		def __init__(self, ID):
			threading.Thread.__init__(self, name = "Staff_%02d" % ID)
			self.ID = ID

		def run(self):
			if take_log: push_info(f"start .. ")
			while True:
				task = manager()
				if task is None: break
				# todo: find a better way
				result = main_func(*task)
				if result_func is None: continue
				thread_lock.acquire()
				if result is None: result_func()
				else: result_func(*result)
				thread_lock.release()
			if take_log: push_info(f"finished .. ")

		pass

	staffs = np.empty(thread_num, object)
	for i in range(thread_num): staffs[i] = Staff(i)
	print("============================================================")
	push_info(f"{title} start, {thread_num} Staff hired ")
	for s in staffs: s.start()
	for s in staffs: s.join()
	push_info(f"{title} finished .. ")
	print("============================================================")
