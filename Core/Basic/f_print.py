from colorama import Fore, Style


def errInfo(info: str):
	return Fore.MAGENTA + info + Style.RESET_ALL


def errPrint(info: str):
	print(errInfo(info))


def sysInfo(info: str):
	return Fore.CYAN + info + Style.RESET_ALL


def sysPrint(info: str):
	print(errInfo(info))


def colInfo(info: str, cn: str):
	"""
	:param info:
	:param cn: terminal basic colors
	:return:
	"""
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
	return style[cn.upper()] + info + Style.RESET_ALL


def colPrint(info: str, cn: str):
	print(colInfo(info, cn))
