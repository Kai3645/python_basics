import logging
import sys

import colorlog as colorlog
from colorama import Fore, Style

LEVEL_DICT = {  # DEBUG < INFO < WARNING < ERROR < CRITICAL
	"DEBUG": logging.DEBUG,
	"INFO": logging.INFO,
	"WARNING": logging.WARNING,
	"ERROR": logging.ERROR,
	"CRITICAL": logging.CRITICAL,
}


class Log(object):
	def __init__(self):
		# create a logger
		self._logger = colorlog.getLogger()
		self._logger.setLevel(logging.INFO)  # default level
		
		# create a stream handler，for console print
		self._sh = colorlog.StreamHandler(sys.stdout)
		self._sh.setLevel(logging.ERROR)  # console level
		self._logger.addHandler(self._sh)
		self._sh.setFormatter(colorlog.ColoredFormatter(
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
		
		# left a file handler，for file print
		self._fh = None
		
		# test logger
		self._logger.debug("logging created ..")
		
		pass
	
	def get_log(self):
		return self._logger
	
	def set_file_handler(self, log_file: str, level: str = "INFO"):
		# create a handler，for file print
		self._fh = logging.FileHandler(log_file)
		level = level.upper()
		if level not in LEVEL_DICT: level = "DEBUG"
		self._fh.setLevel(LEVEL_DICT[level])
		self._logger.addHandler(self._fh)
		self._fh.setFormatter(logging.Formatter(
			"%(levelname).1s %(asctime)s >> %(message)s",
			datefmt = "%Y-%m-%d %H:%M:%S",
		))
		
		# test logger
		self._logger.debug("file handler created ..")
	
	def set_sh_level(self, level: str):
		"""
		:param level: default DEBUG
		:return:
		"""
		level = level.upper()
		if level not in LEVEL_DICT: level = "DEBUG"
		self._sh.setLevel(LEVEL_DICT[level])
		self._logger.info("shift stream handle level into " + level)
	
	def set_fh_level(self, level: str):
		"""
		:param level: default DEBUG
		:return:
		"""
		level = level.upper()
		if level not in LEVEL_DICT: level = "DEBUG"
		self._fh.setLevel(LEVEL_DICT[level])
		self._logger.info("shift file handle level into " + level)


KaisLog = Log()


def color_print(color: str, *args, sep = ' ', end = '\n', file = None):
	style = {
		"BLACK": Fore.BLACK,
		"RED": Fore.RED,
		"GREEN": Fore.GREEN,
		"YELLOW": Fore.YELLOW,
		"BLUE": Fore.BLUE,
		"MAGENTA": Fore.MAGENTA,
		"CYAN": Fore.CYAN,
		"WHITE": Fore.WHITE,
	}
	tmp_str = sep.join(args)
	tmp_str = style[color.upper()] + tmp_str + Style.RESET_ALL
	print(tmp_str, end = end, file = file)


if __name__ == "__main__":
	log = KaisLog.get_log()
	
	log.debug('debug message')
	log.info('info message')
	log.warning('warning message')
	log.error('error message')
	log.critical('critical message')
	
	color_print("green", "a", "b", "v")
