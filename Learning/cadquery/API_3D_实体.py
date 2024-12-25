if __name__ == "__main__":
	import cadquery as cq
	from cadquery.vis import show_object
	
	# 长方体
	show_object(cq.Workplane().box(5, 4, 3))
	
	# 楔体
	# (dx, dy, dz) -> 基础长方体,基础面 x-z,(xmin, xmax) -> 顶面x范围,(ymin, ymax) -> 顶面 y 范围
	show_object(cq.Workplane().wedge(3, 2, 1, 1, 1, 2, 1))
	
	# 球体
	# direct -> up axis, (a1, a2) -> pitch range, a3 -> yaw
	show_object(cq.Workplane().sphere(3, cq.Vector(1, 1, 1), -90, 30, 270))
	
	# 圆柱
	show_object(cq.Workplane().cylinder(1, 3, cq.Vector(1, 1, 1), 270))
	
	# 文本
	show_object(cq.Workplane().text("abc啊", 36, 3, font = "Comic Sans MS",
	                                kind = "regular", halign = "center", valign = "center"))
