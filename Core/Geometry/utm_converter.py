import math
import numpy as np
import utm


class UTMCalculator:
	"""
	Swap latitude longitude with UTM

	- Reference:
	    1, "https://www.movable-type.co.uk/scripts/latlong-utm-mgrs.html"
	    2, "https://vldb.gsi.go.jp/sokuchi/surveycalc/surveycalc/algorithm/bl2xy/bl2xy.htm"
	    3, "https://vldb.gsi.go.jp/sokuchi/surveycalc/surveycalc/algorithm/xy2bl/xy2bl.htm"

	- Core Constants: (WGS84)
	    a: equatorial Radius
	    rf: reciprocal of flattening
	    k0: UTM scale on the central meridian

	- Referent Constants: (WGS84)
	    r: flattening
	    e: eccentricity
	    n_coe: 3rd flattening
	    n[..]: n_coefficient factorial list
	    A: 2πA is the circumference of a meridian

	- Name of Greek letter
	    α: Alpha
	    β: Beta
	    η: Eta
	    ξ: Xi
	    τ: Tau
	    δ: Delta
	    σ: Sigma
	"""
	a = 6378137.0
	rf = 298.257223563
	f = 1 / rf
	k0 = 0.9996
	e = math.sqrt((2 - f) / rf)
	n_coe = 1 / (2 * rf - 1)
	n = np.ones(7)
	for i in range(1, 7): n[i] = n[i - 1] * n_coe
	A = a / (1 + n[1]) * (1 + n[2] / 4 + n[4] / 64 + n[6] / 256)
	A_k0 = A * k0

	# Offset Constants, I don't know what it for .. (@Kai3645)
	falseEast = 5e5
	falseNorth = 1e7

	alpha_coe = np.asarray([  # α is one-based gga_list (6th order Krüger expressions)
		n[1] / 2 - n[2] * 2 / 3 + n[3] * 5 / 16 + n[4] * 41 / 180 - n[5] * 127 / 288 + n[6] * 7891 / 37800,
		n[2] * 13 / 48 - n[3] * 3 / 5 + n[4] * 557 / 1440 + n[5] * 281 / 630 - n[6] * 1983433 / 1935360,
		n[3] * 61 / 240 - n[4] * 103 / 140 + n[5] * 15061 / 26880 + n[6] * 167603 / 181440,
		n[4] * 49561 / 161280 - n[5] * 179 / 168 + n[6] * 6601661 / 7257600,
		n[5] * 34729 / 80640 - n[6] * 3418889 / 1995840,
		n[6] * 212378941 / 319334400
	])

	beta_coe = np.asarray([  # β is one-based gga_list (6th order Krüger expressions)
		n[1] / 2 - n[2] * 2 / 3 + n[3] * 37 / 96 - n[4] / 360 - n[5] * 81 / 512 + n[6] * 96199 / 604800,
		n[2] / 48 + n[3] / 15 - n[4] * 437 / 1440 + n[5] * 46 / 105 - n[6] * 1118711 / 3870720,
		n[3] * 17 / 480 - n[4] * 37 / 840 - n[5] * 209 / 4480 + n[6] * 5569 / 90720,
		n[4] * 4397 / 161280 - n[5] * 11 / 504 - n[6] * 830251 / 7257600,
		n[5] * 4583 / 161280 - n[6] * 108847 / 3991680,
		n[6] * 20648693 / 638668800,
	])

	@classmethod
	def latlon2utm_(cls, lat: float, lon: float):
		# advance calc
		cos_lon = math.cos(lon)
		sin_lon = math.sin(lon)

		Tau = math.tan(lat)
		Sigma = math.sinh(cls.e * math.atanh(cls.e * Tau / math.sqrt(1 + Tau * Tau)))
		Tau1 = Tau * math.sqrt(1 + Sigma * Sigma) - Sigma * math.sqrt(1 + Tau * Tau)
		Xi = math.atan2(Tau1, cos_lon)
		Eta = math.asinh(sin_lon / math.sqrt(Tau1 * Tau1 + cos_lon * cos_lon))

		# numpy gga_list advance calc
		tmp_Num = np.arange(1, 7)
		tmp_sin_Xi = np.sin(2 * Xi * tmp_Num)
		tmp_cos_Xi = np.cos(2 * Xi * tmp_Num)
		tmp_sinh_Eta = np.sinh(2 * Eta * tmp_Num)
		tmp_cosh_Eta = np.cosh(2 * Eta * tmp_Num)

		sum_Xi = np.sum(cls.alpha_coe * tmp_sin_Xi * tmp_cosh_Eta) + Xi
		sum_Eta = np.sum(cls.alpha_coe * tmp_cos_Xi * tmp_sinh_Eta) + Eta

		x = cls.A_k0 * sum_Eta
		y = cls.A_k0 * sum_Xi
		return x, y

	@classmethod
	def latlon2utm(cls, latitudes, longitudes, zone: int = None):
		latitudes = np.atleast_1d(np.array(latitudes))
		longitudes = np.atleast_1d(np.array(longitudes))
		data_len = len(latitudes)

		is_north = latitudes >= 0  # hemisphere
		if zone is None:
			zone = int((longitudes[0] + 180) // 6 + 1)
			# grid zones are 8° tall; 0°N is offset 10 into latitude bands array
			mgrsLatBands = 'CDEFGHJKLMNPQRSTUVWXX'  # X is repeated for 80-84°N
			latBand = mgrsLatBands[int(latitudes[0] // 8 + 10)]
			# adjust zone & central meridian for Norway
			if zone == 31 and latBand == 'V' and longitudes[0] >= 3: zone += 1
			# adjust zone & central meridian for Svalbard
			if zone == 32 and latBand == 'X' and longitudes[0] < 9: zone -= 1
			if zone == 32 and latBand == 'X' and longitudes[0] >= 9: zone += 1
			if zone == 34 and latBand == 'X' and longitudes[0] < 21: zone -= 1
			if zone == 34 and latBand == 'X' and longitudes[0] >= 21: zone += 1
			if zone == 36 and latBand == 'X' and longitudes[0] < 33: zone -= 1
			if zone == 36 and latBand == 'X' and longitudes[0] >= 33: zone += 1

		# offset longitude by longitudinal zone
		longitudes -= (zone - 1) * 6 - 180 + 3

		# Change [deg] into [rad]
		lats = np.deg2rad(latitudes)
		lons = np.deg2rad(longitudes)

		xs = np.zeros(data_len)
		ys = np.zeros(data_len)
		for i, (lat, lon) in enumerate(zip(lats, lons)):
			xs[i], ys[i] = cls.latlon2utm_(lat, lon)

		xs += cls.falseEast
		if ys[0] < 0: ys += cls.falseNorth
		if data_len > 1: return xs, ys, zone, is_north
		return xs[0], ys[0], zone, is_north

	@classmethod
	def utm2latlon_(cls, x, y):
		Eta = x / cls.A_k0
		Xi = y / cls.A_k0

		# numpy gga_list advance calc
		tmp_Num = np.arange(1, 7)
		tmp_sin_Xi = np.sin(2 * Xi * tmp_Num)
		tmp_cos_Xi = np.cos(2 * Xi * tmp_Num)
		tmp_sinh_Eta = np.sinh(2 * Eta * tmp_Num)
		tmp_cosh_Eta = np.cosh(2 * Eta * tmp_Num)

		sum_Xi = Xi - np.sum(cls.beta_coe * tmp_sin_Xi * tmp_cosh_Eta)
		sum_Eta = Eta - np.sum(cls.beta_coe * tmp_cos_Xi * tmp_sinh_Eta)

		# advance calc
		sinh_sum_Eta = math.sinh(sum_Eta)
		sin_sum_Xi = math.sin(sum_Xi)
		cos_sum_Xi = math.cos(sum_Xi)

		Tau1 = sin_sum_Xi / math.sqrt(sinh_sum_Eta * sinh_sum_Eta + cos_sum_Xi * cos_sum_Xi)
		Tau = Tau1
		while True:
			Tau2 = Tau * Tau
			Sigma = math.sinh(cls.e * math.atanh(cls.e * Tau / math.sqrt(1 + Tau2)))
			tmp_sum_Tau = Tau * math.sqrt(1 + Sigma * Sigma) - Sigma * math.sqrt(1 + Tau2)
			delta_Tau = (Tau1 - tmp_sum_Tau) / math.sqrt(1 + tmp_sum_Tau * tmp_sum_Tau)
			delta_Tau *= (1 / (1 - cls.e * cls.e) + Tau2) / math.sqrt(1 + Tau2)
			Tau += delta_Tau

			if abs(delta_Tau) < 1e-12: break

		lat = math.atan(Tau)
		lon = math.atan2(sinh_sum_Eta, cos_sum_Xi)
		return lat, lon

	@classmethod
	def utm2latlon(cls, xs, ys, zone, is_north: bool = True):
		xs = np.atleast_1d(np.array(xs))
		ys = np.atleast_1d(np.array(ys))
		data_len = len(xs)

		xs -= cls.falseEast
		if not is_north: ys -= cls.falseNorth

		lats = np.zeros(data_len)
		lons = np.zeros(data_len)
		for i, (x, y) in enumerate(zip(xs, ys)):
			lats[i], lons[i] = cls.utm2latlon_(x, y)

		latitudes = np.rad2deg(lats)
		longitudes = np.rad2deg(lons)
		longitudes += ((zone - 1) * 6 - 180 + 3)

		if data_len > 1: return latitudes, longitudes
		return latitudes[0], longitudes[0]

	@classmethod
	def info_latlon(cls, lat: float, lon: float):
		x, y, zone, is_north = cls.latlon2utm(lat, lon)
		print(f"lat/lon = {lat:.8f}, {lon:.8f}")
		print(f"utm x, y, zone = {x:.4f}, {y:.4f}, {zone}")
		return x, y, zone, is_north

	@classmethod
	def info_utm(cls, x: float, y: float, zone: int, is_north: bool):
		lat, lon = cls.utm2latlon(x, y, zone, is_north)
		print(f"lat/lon = {lat:.8f}, {lon:.8f}")
		print(f"utm x, y, zone = {x:.4f}, {y:.4f}, {zone}")
		return x, y, zone, is_north

	pass


def LatLon2UTM_accurate(latitude, longitude, zone: int = None):
	"""
	:param latitude:
	:param longitude:
	:param zone:
	:return:
		x:
		y:
		zone:
		is_north:
	"""
	return UTMCalculator.latlon2utm(latitude, longitude, zone)


def UTM2LatLon_accurate(x, y, zone, is_north: bool = True):
	"""
	:param x:
	:param y:
	:param zone:
	:param is_north:
	:return:
		latitude:
		longitude
	"""
	return UTMCalculator.utm2latlon(x, y, zone, is_north)


def LatLon2UTM(latitude, longitude, zone: int = None):
	"""
	:param latitude:
	:param longitude:
	:param zone:
	:return:
		x:
		y:
		zone:
		zone_latter: some bugs here
	"""
	return utm.from_latlon(latitude, longitude, zone)


def UTM2LatLon(x, y, zone, is_north: bool = True):
	"""
	:param x:
	:param y:
	:param zone:
	:param is_north:
	:return:
		latitude:
		longitude
	"""
	return utm.to_latlon(x, y, zone, northern = is_north)


if __name__ == '__main__':
	def test():
		LatLon_test_data = np.asarray([
			[35.33851450, 139.50589728],
			[35.33995992, 139.51000678],
			[35.34047122, 139.51101844],
			[35.34148575, 139.51296194],
			[35.34185600, 139.51362342],
			[35.34256619, 139.51489314],
			[35.34292681, 139.51563047],
			[35.34355986, 139.51631844],
		])

		# for lat, lon in LatLon_test_data:
		# 	print(f"be4 lat/lon = {lat:.8f}, {lon:.8f}")
		# 	x, y, zone, _ = LatLon2UTM(lat, lon)
		# 	print(f"utm x, y, zone = {x:.4f}, {y:.4f}, {zone}")
		# 	lat, lon = UTM2LatLon(x, y, z)
		# 	print(f"aft lat/lon = {lat:.8f}, {lon:.8f}")
		# 	print()

		# lats = LatLon_test_data[:, 0]
		# lons = LatLon_test_data[:, 1]
		# xs, ys, zone, _ = LatLon2UTM(lats, lons)
		# for i, (lat, lon) in enumerate(LatLon_test_data):
		# 	print(f"be4 lat/lon = {lat:.8f}, {lon:.8f}")
		# 	x, y = xs[i], ys[i]
		# 	print(f"utm x, y, zone = {x:.4f}, {y:.4f}, {zone}")
		# 	lat, lon = UTM2LatLon(x, y, zone)
		# 	print(f"aft lat/lon = {lat:.8f}, {lon:.8f}")
		# 	print()

		lats = LatLon_test_data[:, 0]
		lons = LatLon_test_data[:, 1]
		xs, ys, zone, _ = LatLon2UTM(lats, lons)
		lats, lons = UTM2LatLon(xs, ys, zone)
		for i, (lat, lon) in enumerate(LatLon_test_data):
			print(f"be4 lat/lon = {lat:.8f}, {lon:.8f}")
			x, y, zone, _ = LatLon2UTM(lat, lon)
			print(f"utm x, y, zone = {x:.4f}, {y:.4f}, {zone}")
			x, y = xs[i], ys[i]
			print(f"utm x, y, zone = {x:.4f}, {y:.4f}, {zone}")
			lat, lon = lats[i], lons[i]
			print(f"aft lat/lon = {lat:.8f}, {lon:.8f}")
			print()

		print(">> finished ..")
		pass


	test()
