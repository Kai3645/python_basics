from Core.Basic.c_log import Log

NUM_ERROR = 1e-9
NUM_ZERO = 1e-4
TIME_24HOUR = 86400

KaisLog = Log()


def is_zero(x, zero = NUM_ERROR):
	if x < -zero: return False
	if x > zero: return False
	return True
