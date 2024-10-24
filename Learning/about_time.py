if __name__ == '__main__':
	from datetime import datetime, timezone, timedelta
	
	"""
	   ┌───────────> datetime o───────────┐
	   │                                  │
	   │  datetime.strptime()             │
	   │                                  │
	   │  datetime.strftime()             │  datetime.timestamp()
	   │                                  │
	   v      datetime.fromtimestamp()    v
	 string <─────────────────────────o float
	"""
	
	# time datetime
	t_current = datetime.now(tz = timezone(timedelta(hours = 8)))
	
	print("datetime ->", t_current)
	print("string ->", t_current.strftime("%A(%a), %w, %d, %B(%b), %m, %Y(%y), %H, %M, %S, %f, %z, %j"))
	print("float ->", t_current.timestamp())
	print()
	
	# time string
	t_current_str = str(t_current)
	
	print("datetime ->", datetime.strptime(t_current_str, "%Y-%m-%d %H:%M:%S.%f%z"))
	print("string ->", t_current_str)
	print("float ->", )
	print()
	
	# time float
	t_stamp = t_current.timestamp()
	
	print("datetime ->", datetime.fromtimestamp(t_stamp, tz = timezone(timedelta(hours = 8))))
	print("string ->", )
	print("float ->", t_stamp)
	print()
	
	# special day time
	# start of today 00:00:00 utc ("%Y-%m-%d %H:%M:%S %z")
	t_day_start_str = t_current.strftime("%Y-%m-%d %z")
	t_day_start = datetime.strptime(t_day_start_str, "%Y-%m-%d %z")
	print("start of today =", t_day_start.strftime("%Y-%m-%d %A %H:%M:%S.%f %z"))
	
	# start of week Monday 00:00:00.0 utc ("%Y %W %w %H:%M:%S %z")
	t_week_start_str = t_current.strftime("%Y %W 1 %z")
	t_week_start = datetime.strptime(t_week_start_str, "%Y %W %w %z")
	print("start of week =", t_week_start.strftime("%Y-%m-%d %A %H:%M:%S.%f %z"))
	
	# start of month 00:00:00.0 utc ("%Y-%m-%d %H:%M:%S %z")
	t_month_start_str = t_current.strftime("%Y-%m-01 %z")
	t_month_start = datetime.strptime(t_month_start_str, "%Y-%m-%d %z")
	print("start of month =", t_month_start.strftime("%Y-%m-%d %A %H:%M:%S.%f %z"))
	
	# start of year 00:00:00.0 utc ("%Y-%m-%d %H:%M:%S %z")
	t_year_start_str = t_current.strftime("%Y-01-01 %z")
	t_year_start = datetime.strptime(t_year_start_str, "%Y-%m-%d %z")
	print("start of year =", t_year_start.strftime("%Y-%m-%d %A %H:%M:%S.%f %z"))
	
	
	# time calculate
	def the_nth_weekday_of_month(d_month_start: datetime, n, weekday):
		"""
		calculate the Nth weekday of the month
		:param d_month_start: date of the month start
		:param n:
		:param weekday: Tuesday = 2
		:return:
		"""
		weekday_0 = d_month_start.isoweekday()
		if weekday_0 > weekday: d_delta = timedelta(days = n * 7 + weekday - weekday_0)
		else: d_delta = timedelta(days = (n - 1) * 7 + weekday - weekday_0)
		return d_month_start + d_delta
	
	
	t_month_start = datetime.strptime("2024-08-01 +0800", "%Y-%m-%d %z")
	print("2024-08 2nd tuesday =",
	      the_nth_weekday_of_month(t_month_start, 2, 2).strftime(
		      "%Y-%m-%d %A %H:%M:%S.%f %z"
	      ))
	pass
