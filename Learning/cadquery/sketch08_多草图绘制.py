if __name__ == "__main__":
	import cadquery as cq
	from cadquery.vis import show_object
	
	result = (
		cq.Workplane()
		.box(5, 5, 1)
		.faces(">Z")
		.workplane()
		.rarray(2, 2, 2, 2)  # 阵列
		.rect(1.5, 1.5)
		.extrude(0.5)  # 拉伸
		.faces(">Z")
		.sketch()
		.circle(0.4)
		.wires()
		.distribute(6)
		.circle(0.1, mode = "a")
		.clean()  # 清理
		.finalize()  # 完成
		.cutBlind(-0.5, taper = 10)  # 盲孔-带锥度
	)
	show_object(result)
