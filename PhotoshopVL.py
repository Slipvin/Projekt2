from PIL import ImageEnhance
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image, ImageDraw

s = int(input('Choose(0-6) :'))
image = Image.open("nebo.jpg")
draw = ImageDraw.Draw(image)
width = image.size[0]
height = image.size[1]
pix = image.load()

if (s == 0):
	for i in range(width):
		for j in range(height):
			x = pix[i, j][0]
			y = pix[i, j][1]
			z = pix[i, j][2]
			S = (x + y + z) // 3
			draw.point((i, j), (S, S, S))

if (s == 1):
	depth = int(input('depth:'))
	for i in range(width):
		for j in range(height):
			x = pix[i, j][0]
			y = pix[i, j][1]
			z = pix[i, j][2]
			S = (x + y + z) // 3
			x = S + depth * 2
			y = S + depth
			z = S
			if (x > 255):
				x = 255
			if (y > 255):
				y = 255
			if (z > 255):
				z = 255
			draw.point((i, j), (x, y, z))

if (s == 2):
	for i in range(width):
		for j in range(height):
			x = pix[i, j][0]
			y = pix[i, j][1]
			z = pix[i, j][2]
			draw.point((i, j), (255 - x, 255 - y, 255 - z))

if (s == 3):
	factor = int(input('factor:'))
	for i in range(width):
		for j in range(height):
			rand = random.randint(-factor, factor)
			x = pix[i, j][0] + rand
			y = pix[i, j][1] + rand
			z = pix[i, j][2] + rand
			if (x < 0):
				x = 0
			if (y < 0):
				y = 0
			if (z < 0):
				z = 0
			if (x > 255):
				x = 255
			if (y > 255):
				y = 255
			if (z > 255):
				z = 255
			draw.point((i, j), (x, y, z))

if (s == 4):
	factor = int(input('factor:'))
	for i in range(width):
		for j in range(height):
			x = pix[i, j][0] + factor
			y = pix[i, j][1] + factor
			z = pix[i, j][2] + factor
			if (x < 0):
				x = 0
			if (y < 0):
				y = 0
			if (z < 0):
				z = 0
			if (x > 255):
				x = 255
			if (y > 255):
				y = 255
			if (z > 255):
				z = 255
			draw.point((i, j), (x, y, z))

if (s == 5):
	factor = int(input('factor:'))
	for i in range(width):
		for j in range(height):
			x = pix[i, j][0]
			y = pix[i, j][1]
			z = pix[i, j][2]
			S = x + y + z
			if (S > (((255 + factor) // 2) * 3)):
				x, y, z = 255, 255, 255
			else:
				x, y, z = 0, 0, 0
			draw.point((i, j), (x, y, z))

if (s == 6):
	im = Image.open("nebo.jpg")

	enhancer = ImageEnhance.Contrast(im)

	factor = 1
	im_output = enhancer.enhance(factor)
	im_output.save('original-image.png')

	factor = 0.5
	im_output = enhancer.enhance(factor)
	im_output.save('less-contrast-image.png')

	factor = 1.5
	im_output = enhancer.enhance(factor)
	im_output.save('more-contrast-image.png')

	img = cv2.imread('nebo.jpg')
	img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	gaussianBlur = cv2.GaussianBlur(grayImage, (3, 3), 0)

	ret, binary = cv2.threshold(gaussianBlur, 127, 255, cv2.THRESH_BINARY)

	kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
	kernely = np.array([[0, -1], [1, 0]], dtype=int)
	x = cv2.filter2D(binary, cv2.CV_16S, kernelx)
	y = cv2.filter2D(binary, cv2.CV_16S, kernely)
	absX = cv2.convertScaleAbs(x)
	absY = cv2.convertScaleAbs(y)
	Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

	kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
	kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
	x = cv2.filter2D(binary, cv2.CV_16S, kernelx)
	y = cv2.filter2D(binary, cv2.CV_16S, kernely)
	absX = cv2.convertScaleAbs(x)
	absY = cv2.convertScaleAbs(y)
	Prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

	x = cv2.Sobel(binary, cv2.CV_16S, 1, 0)
	y = cv2.Sobel(binary, cv2.CV_16S, 0, 1)
	absX = cv2.convertScaleAbs(x)
	absY = cv2.convertScaleAbs(y)
	Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

	dst = cv2.Laplacian(binary, cv2.CV_16S, ksize=3)
	Laplacian = cv2.convertScaleAbs(dst)

	plt.subplot(231), plt.imshow(img_RGB), plt.axis('off')
	plt.subplot(232), plt.imshow(gaussianBlur, cmap=plt.cm.gray), plt.axis('off')
	plt.subplot(233), plt.imshow(Roberts, cmap=plt.cm.gray), plt.axis('off')
	plt.subplot(234), plt.imshow(Prewitt, cmap=plt.cm.gray), plt.axis('off')
	plt.subplot(235), plt.imshow(Sobel, cmap=plt.cm.gray),  plt.axis('off')
	plt.subplot(236), plt.imshow(Laplacian, cmap=plt.cm.gray), plt.axis('off')

	plt.show()

image.save("result.jpg", "JPEG")
del draw