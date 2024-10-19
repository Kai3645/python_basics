"""
This file storage constant scalars
and similar functions
"""

NUM_ERROR = 1e-9  # set minimum number for float64
NUM_ZERO = 1e-4  # Set minimum number that can be recognized as zero in real life

TIME_HOUR = 3600  # [second]
TIME_24HOUR = 86400  # [second]


def is_zero(x, min_num = NUM_ERROR):
	if x < -min_num: return False
	if x > min_num: return False
	return True
