from datetime import datetime, date, timezone


# todo: update when needed
# def weekTimestamp(dt: datetime):
# 	"""
# 	dangerous time zone @ Saturday 12:00:00 ~ Sunday 12:00:00
# 	:param dt: datetime @ timezone +0.0
# 	:return:
# 	"""
# 	t = ((dt.weekday() + 1) % 7) * 86400
# 	t += dt.hour * 3600
# 	t += dt.minute * 60
# 	t += dt.second
# 	t += dt.microsecond * 1e-6
# 	return t

# todo: ValueError: day is out of range for month
# def weekTimestamp_recover(t: float, d: date):
# 	dt = datetime(year = d.year, month = d.month, day = d.day - d.weekday() + 1)
# 	return datetime.fromtimestamp(dt.timestamp() + t, timezone.utc)


def timestamp2str(t: float, timeFormat: str = "%Y%m%d%H%M%S%f", tz = timezone.utc):
	dt = datetime.fromtimestamp(t, tz)
	return dt.strftime(timeFormat)


def str2timestamp(t_str: str, timeFormat: str = "%Y%m%d%H%M%S%f", tz = timezone.utc):
	dt = datetime.strptime(t_str, timeFormat)
	dt = dt.replace(tzinfo = tz)
	return round(dt.timestamp(), 6)
