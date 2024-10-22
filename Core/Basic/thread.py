import queue
import threading

import numpy as np

from Core.Basic.log import KaisLog

log = KaisLog.get_log()


def multithreading(task_list, main_func, result_func, *, title = "multi", thread_num: int = 10, CD_step: int = 10):
	"""
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
	:param CD_step: countdown for log default = 10
	:return: None
	"""
	total_jobs = len(task_list)
	thread_num = min(thread_num, total_jobs)
	CD_step = min(total_jobs, CD_step)
	jobs = queue.Queue(total_jobs)
	for w in task_list: jobs.put(w, block = True)
	thread_lock = threading.Lock()
	
	def push_info(info):
		tmp_str = title + " | "
		tmp_str += threading.current_thread().name + " >> " + info
		thread_lock.acquire()
		log.info(tmp_str)
		thread_lock.release()
	
	def manager():
		if jobs.qsize() == 0: return None
		task = jobs.get(block = True)
		jobs.task_done()
		num = jobs.qsize()
		if num % CD_step == 0:
			rate = num / total_jobs * 100
			push_info(f"------------ {rate:.2f} % ------------")
		return task
	
	class Staff(threading.Thread):
		def __init__(self, ID):
			threading.Thread.__init__(self, name = "Staff_%02d" % ID)
			self.ID = ID
		
		def run(self):
			push_info(f"start .. ")
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
			push_info(f"finished .. ")
		
		pass
	
	staffs = np.empty(thread_num, object)
	for i in range(thread_num): staffs[i] = Staff(i)
	log.info("============================================================")
	log.info(f"{title} start, {thread_num} Staff hired ")
	for s in staffs: s.start()
	for s in staffs: s.join()
	log.info(f"{title} finished .. ")
	log.info("============================================================")


if __name__ == '__main__':
	KaisLog.set_sh_level("info")
	
	# math mat test
	M, N = 100, 10000000
	
	mat = np.random.random((M, N))
	ret = np.zeros((M, M))
	
	_task_list = [(i, mat[i], mat) for i in range(M)]
	
	
	def _main_func(_idx, _arr, _mat):
		_ret = []
		m, n = _mat.shape
		for _i in range(m):
			_ret.append(np.sum(_arr * _mat[_i]))
		return _idx, _ret
	
	
	def _result_func(_idx, _arr):
		ret[_idx, :] = _arr
		pass
	
	
	multithreading(_task_list, _main_func, _result_func, thread_num = 20, CD_step = 4)
	# mark: thread_num(8) is faster than thread_num(20) ???
	
	# count = 1
	# for _t in _task_list:
	# 	if count % 4 == 0:
	# 		rate = count / len(_task_list) * 100
	# 		log.info(f"------------ {rate:.2f} % ------------")
	# 	tmp = _main_func(*_t)
	# 	_result_func(*tmp)
	
	pass
