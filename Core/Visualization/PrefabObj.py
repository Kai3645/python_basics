import math

import numpy as np
import openmesh as om

from Core.Basic import is_zero, KaisLog

log = KaisLog.get_log()


class PrefabObj:
	@staticmethod
	def Cuboid(a, b, c):
		points = np.asarray([
			[a, b, c], [-a, b, c], [-a, -b, c], [a, -b, c],
			[a, b, -c], [-a, b, -c], [-a, -b, -c], [a, -b, -c],
		], np.float64) / 2
		fv_idxes = np.asarray([
			[0, 1, 2, 3], [0, 3, 7, 4], [0, 4, 5, 1],
			[1, 5, 6, 2], [2, 6, 7, 3], [5, 4, 7, 6],
		], np.int32)
		return om.PolyMesh(points, fv_idxes)

	@classmethod
	def Cube(cls, a):
		return cls.Cuboid(a, a, a)

	@classmethod
	def RegularTetrahedron(cls, r):
		a = -r / 3
		b = -r * math.sqrt(2) / 3
		c = -b * 2
		e = r * math.sqrt(6) / 3
		points = np.asarray([
			[r, 0, 0],
			[a, c, 0],
			[a, b, e],
			[a, b, -e],
		], np.float64)
		fv_idxes = np.asarray([[0, 1, 2], [0, 2, 3], [0, 3, 1], [3, 2, 1]], np.int32)
		return om.PolyMesh(points, fv_idxes)

	@staticmethod
	def Camera(w, h, f):
		w_ = w / 2
		h_ = h / 2
		points = np.asarray([
			[0, 0, 0],
			[w_, h_, f], [w_, -h_, f],
			[-w_, -h_, f], [-w_, h_, f],
		], np.float64)
		fv_idxes = np.asarray([
			[2, 1, 4, 3], [0, 1, 2, -1], [0, 2, 3, -1],
			[0, 3, 4, -1], [0, 4, 1, -1],
		], np.int32)
		return om.PolyMesh(points, fv_idxes)

	@staticmethod
	def Dodecahedron(r):
		sqrt_5 = math.sqrt(5)
		a = (sqrt_5 + 1) / 2
		b = (sqrt_5 - 1) / 2
		points = np.asarray([
			[0, a, b], [0, -a, b], [0, -a, -b], [0, a, -b],
			[b, 0, a], [-b, 0, a], [-b, 0, -a], [b, 0, -a],
			[a, b, 0], [-a, b, 0], [-a, -b, 0], [a, -b, 0],
			[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1],
			[1, 1, -1], [-1, 1, -1], [-1, -1, -1], [1, -1, -1],
		], np.float64) * (r / math.sqrt(3))
		fv_idxes = np.asarray([
			[0, 13, 5, 4, 12], [0, 12, 8, 16, 3],
			[0, 3, 17, 9, 13], [13, 9, 10, 14, 5],
			[5, 14, 1, 15, 4], [4, 15, 11, 8, 12],
			[8, 11, 19, 7, 16], [16, 7, 6, 17, 3],
			[17, 6, 18, 10, 9], [10, 18, 2, 1, 14],
			[1, 2, 19, 11, 15], [19, 2, 18, 6, 7],
		], np.int32)
		return om.PolyMesh(points, fv_idxes)

	@staticmethod
	def refinement(src):
		dst = om.PolyMesh()
		pts = src.points()
		he2v_idxes = src.halfedge_vertex_indices()
		he2vh_dst = np.empty(len(he2v_idxes), object)
		for fh in src.faces():
			fc = src.calc_face_centroid(fh)
			vhs = []
			for heh in src.fh(fh):
				he_id = heh.idx()
				v_id1, v_id2 = he2v_idxes[he_id]
				x = (pts[v_id1] + pts[v_id2] + fc) / 3
				vh_dst = dst.add_vertex(x)
				he2vh_dst[he_id] = vh_dst
				vhs.append(vh_dst)
			dst.add_face(vhs)
		he_dst2v_dst_idxes = dst.halfedge_vertex_indices()
		for vh in src.vertices():
			vhs = []
			handles = [heh for heh in src.vih(vh)]
			for heh in handles[::-1]:
				ih = src.opposite_halfedge_handle(heh)
				vh_dst = he2vh_dst[ih.idx()]
				vhs.append(vh_dst)
				he_dst_id = next(dst.voh(vh_dst)).idx()
				v_dst_id = he_dst2v_dst_idxes[he_dst_id][1]
				vh_dst = dst.vertex_handle(v_dst_id)
				vhs.append(vh_dst)
			dst.add_face(vhs)
		return dst

	@staticmethod
	def sphere_optimize(mesh, r, limit_dist = 4e-5, limit_loop = 100):
		ff_ids = mesh.face_face_indices()
		face_count = len(ff_ids)
		fc_data = np.zeros((face_count, 3))
		for fh in mesh.faces():
			fc = mesh.calc_face_centroid(fh)
			fc_data[fh.idx(), :] = fc / np.linalg.norm(fc)

		def fc_offset(f0, pts):
			# better face size, but looks bad
			# _len = len(pts)
			# pts = pts - f0
			# _vs = [_pi - pts[_i + 1 - _len] for _i, _pi in enumerate(pts)]
			# _ls = [np.dot(_vi, _vi) for _vi in _vs]
			# _ks = [_li + _ls[_i - 1] for _i, _li in enumerate(_ls)]
			# _fs = [_ki * _pi for _ki, _pi in zip(_ks, pts)]
			# return np.sum(_fs, axis = 0) / np.sum(_ks)

			assert not is_zero(np.sum(np.abs(f0)))
			if not is_zero(f0[0]):
				den = math.sqrt(f0[0] * f0[0] + f0[1] * f0[1])
				_W = np.asarray([[-f0[1] / den, f0[0] / den, 0], [-f0[2], 0, f0[0]]])
			elif not is_zero(f0[1]):
				den = math.sqrt(f0[0] * f0[0] + f0[1] * f0[1])
				_W = np.asarray([[f0[1] / den, -f0[0] / den, 0], [0, -f0[2], f0[1]]])
			else:
				den = math.sqrt(f0[0] * f0[0] + f0[2] * f0[2])
				_W = np.asarray([[f0[2] / den, 0, -f0[0] / den], [0, f0[2], -f0[1]]])
			_W[1] -= np.dot(_W[1], _W[0]) * _W[0]
			_W[1] /= np.linalg.norm(_W[1])
			_Wt = _W.transpose()
			_T = np.dot(_Wt, np.linalg.inv(np.dot(_W, _Wt)))
			pts = np.dot(pts, _T)

			_A = np.asarray([[_x, _y, 1] for _x, _y in pts])
			_At = _A.transpose()
			_T = np.dot(np.linalg.inv(np.dot(_At, _A)), _At)
			_B = [_x * _x + _y * _y for _x, _y in pts]
			_a = np.dot(_T[0], _B) * 0.75
			_b = np.dot(_T[1], _B) * 0.75
			return _W[0] * _a + _W[1] * _b

		loop = 0
		err_max, err_pre = 1, 0
		while err_max > limit_dist:
			err_max, err = 0, 0
			for i, f_ids in enumerate(ff_ids):
				if f_ids[-1] < 0: continue
				fcs = fc_data[f_ids]
				delta = fc_offset(fc_data[i], fcs)
				err = sum(np.abs(delta))
				if err > err_max: err_max = err
				f_fix = fc_data[i] + delta
				fc_data[i] = f_fix / np.linalg.norm(f_fix)
			log.info(f"optimize {loop:03d}, err = {err_max - limit_dist:.4e}")
			if is_zero(err_pre - err): break
			if loop > limit_loop: break
			err_pre = err
			loop += 1
		points = mesh.points()
		for vh in mesh.vertices():
			f_ids = [fh.idx() for fh in mesh.vf(vh)]
			fns = fc_data[f_ids[:3]]
			inv_A = np.linalg.inv(fns)
			points[vh.idx(), :] = np.sum(inv_A, axis = 1)
		points *= r
		return mesh

	@classmethod
	def HexagonalSphere(cls, r, depth = 2, do_optimize = False):
		"""
		:param r: incenter sphere radius
		:param depth:
		:param do_optimize:
		:return:
		"""
		assert depth > 0
		if do_optimize: assert depth <= 8
		mesh = PrefabObj.Dodecahedron(1)
		for _ in range(depth): mesh = cls.refinement(mesh)
		if do_optimize: mesh = cls.sphere_optimize(mesh, r)
		else: mesh = cls.sphere_optimize(mesh, r, 1)
		return mesh


if __name__ == '__main__':
	folder = "/home/kai/PycharmProjects/pyCenter/d_2022_0727/out/"


	def main():
		# mesh = PrefabObj.Cuboid(5, 3, 1)
		# mesh = PrefabObj.Cube(4 / math.sqrt(3))
		# mesh = PrefabObj.RegularTetrahedron(2)
		# mesh = PrefabObj.Camera(1.6, 0.9, 1.2)
		# mesh = PrefabObj.Dodecahedron(2)
		# mesh = PrefabObj.HexagonalSphere(2, 6, True)
		# for _ in range(6): mesh = PrefabObj.refinement(mesh)
		# mesh = PrefabObj.sphere_optimize(mesh, 2)
		# om.write_mesh(folder + "sample.obj", mesh)

		# fr = open(folder + "sample.obj", "r")
		# fw = open(folder + "sample_wire.obj", "w")
		# for line in fr:
		# 	if line[0] == "v":
		# 		fw.write(line)
		# 		continue
		# 	if line[0] == "f":
		# 		ids = line[2:-1].split(" ")
		# 		ids.append(ids[0])
		# 		fw.write("l " + " ".join(ids) + "\n")
		# 		continue
		# fr.close()
		# fw.close()
		pass


	main()
	pass
