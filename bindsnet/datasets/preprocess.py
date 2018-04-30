import cv2

def gray_scale(im):
	'''
	Converts RGB image into grayscale.
	
	Inputs:
		im(numpy.array): RGB image.
	
	Returns:
		im(numpy.array): Grayscaled image (All pixel values are from [0, 1])
	'''
	im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
	return im


def crop(im, x1, x2, y1, y2):
	return im[x1:x2, y1:y2, :]
	

def binary_image(im):
	'''
	Converts input image into black and white (binary)
	
	Inputs:
		im(numpy.array): Grayscaled image.
	
	Returns:
		im(numpy.array): Black and white image. 
	'''
	ret, im = cv2.threshold(im, 1, 255, cv2.THRESH_BINARY)
	return im


def subsample(im, x, y):
	'''
	Scale the image to (x, y).
	
	Inputs:
		im(numpy.array): Image to be rescaled.
		x(int): Output value for im's x dimension.
		y(int): Output value for im's y dimension.
	
	Returns:
		im(numpy.array): Rescaled image. 
	'''
	im = cv2.resize(im, (x, y))
	return im 
						 
