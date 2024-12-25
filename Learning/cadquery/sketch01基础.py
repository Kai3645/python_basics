if __name__ == "__main__":
	import cadquery as cq
	from cadquery.vis import show_object
	
	"""
	mode type:
	    a: 融合
	    s: 切割
	    i: 相交
	    r: 替换
	    c: 仅存储以进行构造
	"""
	result = (
		cq.Sketch()
		.trapezoid(4, 3, 90)  # 梯形
		.vertices()  # 取顶点
		.circle(0.5, mode = "s")  # 圆-切割
		.reset()  # 释放选取
		.vertices()  # 取顶点
		.fillet(0.25)  # 圆角
		.reset()  # 释放选取
		.rarray(0.6, 1, 5, 1)  # 矩形阵列,基于中心点
		.slot(1.6, 0.4, mode = "s", angle = 75)  # 圆角曹,基于x轴
	)
	show_object(result)
