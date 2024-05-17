import logging
import sys

import colorlog as colorlog

"""
# ---------- sample ----------
from Core.Basic import KaisLog

KaisLog.set_file_handler(path)
KaisLog.set_level("DEBUG")
log = KaisLog.get_log()

log.debug('debug message')
log.info('info message')
log.warning('warning message')
log.error('error message')
log.critical('critical message')
"""


class Log(object):
	def __init__(self):
		"""
		指定保存日志的文件路径，日志级别，以及调用文件
		将日志存入到指定的文件中
		"""
		# 创建一个logger
		self.logger = colorlog.getLogger()
		self.logger.setLevel(logging.DEBUG)  # main level
		# 创建一个handler，用于输出到控制台
		# create console handler with a higher log level
		self._ch = colorlog.StreamHandler(sys.stdout)
		self._ch.setLevel(logging.ERROR)
		self.logger.addHandler(self._ch)
		self._ch.setFormatter(colorlog.ColoredFormatter(
			"%(asctime)s %(log_color)s%(levelname).1s >> %(message)s",
			datefmt = "%H:%M:%S",
			log_colors = {
				'DEBUG': 'cyan',
				'INFO': 'green',
				'WARNING': 'yellow',
				'ERROR': 'red',
				'CRITICAL': 'purple',
			},
		))

		# test logger
		self.logger.info("logging start")

	def get_log(self):
		return self.logger

	def set_file_handler(self, log_path):
		# 创建一个handler，用于写入日志文件
		# create file handler which logs even debug messages
		fh = logging.FileHandler(log_path)
		fh.setLevel(logging.DEBUG)
		self.logger.addHandler(fh)
		fh.setFormatter(logging.Formatter(
			"%(levelname).1s %(asctime)s >> %(message)s",
			datefmt = "%Y-%m-%d %H:%M:%S",
		))
		# test logger
		self.logger.info("file handler created")

	def set_level(self, level_name: str):
		"""
		DEBUG < INFO < WARNING < ERROR < CRITICAL
		:param level_name:
		:return:
		"""
		name = level_name.upper()
		level_dict = {
			"DEBUG": logging.DEBUG,
			"INFO": logging.INFO,
			"WARNING": logging.WARNING,
			"ERROR": logging.ERROR,
			"CRITICAL": logging.CRITICAL,
		}
		if name not in level_dict: name = "INFO"
		self._ch.setLevel(level_dict[name])
		self.logger.info("shift logging level into " + name)
