import os
import sys

if __name__ == "__main__":
	import math
	
	import cadquery as cq
	from cadquery.occ_impl import exporters
	from cadquery.vis import show_object
	
	folder = os.path.dirname(__file__) + os.path.sep
	print(f"out put path: \"{folder}..\"")
	
	_D_outer = float(sys.argv[1])
	_Height = float(sys.argv[2])
	_Thickness = float(sys.argv[3])
	_R0 = float(sys.argv[4])
	_MAX_DX = 6 if len(sys.argv) < 6 else float(sys.argv[5])
	
	
	def main(D_outer = 28., Height = 50., Thickness = 1.6, R0 = 2., MAX_DX = 6):
		R0 = R0 if R0 > Thickness else Thickness
		X_outer = D_outer / 2
		
		X_inner = X_outer - Thickness
		Ax = X_inner / 2 * 3 - R0
		is_small = False
		if Ax + R0 - X_outer >= MAX_DX:
			Ax = X_outer + MAX_DX - R0
		if Ax < R0 + X_inner:
			is_small = True
			Ax = R0 + X_inner
		
		print(f"D_outer = {D_outer}, D_max = {Ax * 2 + R0 * 2}")
		print(Ax)
		
		F1x = Ax * 2 - X_outer + R0
		F1y = (Ax - X_outer + R0) * math.sqrt(3)
		F1r = (Ax - X_outer) * 2 + R0
		print(F1x, F1r)
		print(F1y)
		
		if not is_small:
			tmp_d = math.sqrt((Ax - X_inner - R0) * (Ax - X_inner - R0) + F1y * F1y)
			F2r = tmp_d * tmp_d / 2 / (Ax - X_inner - R0) + R0
			F2x = X_inner + F2r
			tmp_a = math.atan((Ax - X_inner - R0) / F1y) / math.pi * 360
			print(F2r, F2x)
			S = (
				cq.Sketch()
				.segment((X_inner, 25.0), (X_inner, F1y))
				.segment((X_outer, F1y), (X_outer, 25.0))
				.close()
				.arc((Ax, 0), R0, -180 + tmp_a, 180 - tmp_a + 60)
				.arc((F1x, F1y), F1r, -180, 60)
				.arc((F2x, F1y), F2r, -180, tmp_a)
				.assemble()
				.moved(cq.Location((0, -Height / 2 + R0, 0)))
			)
		# show_object(S)
		
		else:
			S = (
				cq.Sketch()
				.segment((X_inner, 25.0), (X_inner, 0))
				.segment((X_outer, F1y), (X_outer, 25.0))
				.close()
				.arc((Ax, 0), R0, -180, 180 + 60)
				.arc((F1x, F1y), F1r, -180, 60)
				.assemble()
				.moved(cq.Location((0, -Height / 2 + R0, 0)))
			)
		# show_object(S)
		
		body0 = (
			cq.Workplane("XY")
			.placeSketch(S)
			.revolve(360, (0, 0, 0), (0, 1, 0))
		)
		body = body0 + body0.mirror("XZ")
		exporters.export(body, folder + f"ThroughHole_D{D_outer:.1f}_H{Height:.1f}_T{Thickness:.1f}_R{R0:.1f}.step")
		show_object(body)
	
	
	# for r in range(6, 16):
	# 	main(r * 2)
	main(_D_outer, _Height, _Thickness, _R0, _MAX_DX)
	pass
