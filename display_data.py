import matplotlib.pyplot as plt
import numpy as np



def display_data(x):
	(m,n) = x.shape


	width = np.round(np.sqrt(n)).astype(int)
	height = (n / width).astype(int)


	rows = np.floor(np.sqrt(m)).astype(int)
	cols = np.ceil(m / rows).astype(int)


	pad = 1

	display_array = -np.ones((pad + rows*(height+pad),
							  pad + cols*(width + pad)))

	current_image = 0
	for j in range(rows):
		for i in range(cols):
			if current_image > m:
				break
			
			max_val = np.max(np.abs(x[current_image,:]))
			display_array[pad + j*(height + pad) + np.arange(height),
						  pad + i*(width + pad) + np.arange(width)[:,np.newaxis]] = \
						  x[current_image,:].reshape((height,width)) / max_val
			current_image += 1
		if current_image > m :
			break

	# 显示图像
	plt.figure()
	# 设置图像色彩为灰度值，指定图像坐标范围
	plt.imshow(display_array,cmap = 'gray',extent =[-1,1,-1,1])
	plt.axis('off')
	plt.title('Random Seleted Digits')
	plt.show()

