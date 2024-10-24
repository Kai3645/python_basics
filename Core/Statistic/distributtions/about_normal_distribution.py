import math

import numpy as np
from scipy import stats

from Core.Basic import KaisLog, NUM_ERROR

log = KaisLog.get_log()


def verify_normal_distribution(data):
	"""
	work well for researching, doesn't work as a function
	:param data:
	:return:
	"""
	methods_counts = 4
	trust_count = 0
	mu, sigma = 0, -1
	log.info("verify normal distribution:")
	
	# 1, D'Agostino-Pearson:
	if len(data) > 20:
		stat, p = stats.normaltest(data)
		if p > 0.05:
			log.info("\tAgostino: agree")
			trust_count += 2
		else:
			log.info("\tAgostino: disagree")
	else: methods_counts -= 1
	
	# 2, Kolmogorov-Smirnov:
	stat, p = stats.kstest(data - np.mean(data), "norm")
	if p < 0.05:
		log.info("\tKolmogorov: agree")
		trust_count += 2
	else:
		log.info("\tKolmogorov: disagree")
	
	# 3, Shapiro-Wilk, data N should < 5000
	if len(data) < 5000:
		stat, p = stats.shapiro(data)
		if p > 0.05:
			log.info("\tShapiro: agree")
			trust_count += 2
		else:
			log.info("\tShapiro: disagree")
	else: methods_counts -= 1
	
	# 4, Anderson-Darling: compare to critical at 5%, N should > 10
	if len(data) > 10:
		result = stats.anderson(data)
		print(result)
		if result.statistic < result.critical_values[2]:
			log.info("\tAnderson: agree")
			trust_count += 3
			mu += result.fit_result.params[0]
			sigma = result.fit_result.params[1]
		else:
			log.info("\tAnderson: disagree")
	else: methods_counts -= 1
	if trust_count > methods_counts + 1: return True, mu, sigma
	return False, mu, sigma


def fit_normal_distribution(arr, *, level: int = 9, scale: float = -1):
	"""
	based on RANSAC, for more accuracy set scale manually
	:param arr: (n, ) array like
	:param level: int
	:param scale: int, > 2
	:return:
	"""
	n = len(arr)
	if n <= 10: raise ValueError("Not enough data to fit normal distribution")
	if level > 9: level = 9
	if level < 0: level = 0
	cases = (
		3.2757,  # p0 = 0.9990
		3.3038,  # p1 = 0.9991
		3.3347,  # p2 = 0.9992
		3.3692,  # p3 = 0.9993
		3.4082,  # p4 = 0.9994
		3.4532,  # p5 = 0.9995
		3.5065,  # p6 = 0.9996
		3.5722,  # p7 = 0.9997
		3.6582,  # p8 = 0.9998
		3.7846,  # p9 = 0.9999
	)
	if scale < 2: scale = cases[level]
	
	arr = np.asarray(arr)
	
	mean = np.mean(arr)
	idxes = np.argsort(np.abs(arr - mean))
	delta = arr - np.mean(arr)
	sigma = math.sqrt(np.var(delta))
	
	best_id = idxes[0]
	old_sigma = 0
	CD = 20
	
	while idxes[0] != best_id or abs(sigma - old_sigma) > NUM_ERROR:
		delta = arr - arr[best_id]
		idxes = np.argsort(np.abs(delta))
		valid = np.abs(delta) < scale * sigma
		mean = np.mean(arr[valid])
		delta = arr[valid] - mean
		old_sigma = sigma
		sigma = math.sqrt(np.var(delta))
		CD -= 1
		if CD == 0: break
	
	return mean, sigma, idxes[0]


if __name__ == '__main__':
	KaisLog.set_sh_level("info")
	
	
	def main():
		from Core.Visualization.KaisCanvas import KaisCanvas
		f_out = "/home/lab/Desktop/python_resource/M24_10/D2410_23/out/"
		canvas = KaisCanvas(dark_mode = True, fig_size = (12, 8))
		total = 200000
		# create normal distribution, 均值/期望 μ = 2, 标准差 σ = 1
		data = np.random.normal(0.254, 1.945, total)
		dear = np.random.normal(0.8, 10, max(2, int(total / 20)))
		data = np.concatenate((dear, data))
		dear = np.random.normal(-0.4, 4, max(2, int(total / 10)))
		data = np.concatenate((dear, data))
		
		mu2, sigma2, best_id = fit_normal_distribution(data, scale = 3.1)
		print(mu2, sigma2)
		
		delta = data - mu2
		valid = np.abs(delta) > 3 * sigma2
		
		flag, mu, sigma = verify_normal_distribution(data[np.logical_not(valid)])
		if flag: log.info("data is normal distribution")
		else: log.info("data clearly not normal distribution")
		if sigma > 0: print(mu, sigma)
		mu1 = np.mean(data)
		sigma1 = math.sqrt(np.var(data - mu))
		print(mu1, sigma1)
		
		canvas.ax.hist(data[np.logical_not(valid)], histtype = "step", align = "mid", density = True)
		
		T = np.linspace(-20, 20, num = 4000)
		pdf_fitted = getattr(stats, "norm").pdf(T, loc = mu2, scale = sigma2)
		canvas.ax.plot(T, pdf_fitted, color = "yellow")
		
		T = np.linspace(-20, 20, num = 4000)
		pdf_fitted = getattr(stats, "norm").pdf(T, loc = mu1, scale = sigma1)
		canvas.ax.plot(T, pdf_fitted, color = "green")
		
		canvas.set_axis(equal_axis = False, xlim = (-15, 15))
		canvas.save(f_out + "out.jpg", dpi = 300)
	
	
	main()
	pass
