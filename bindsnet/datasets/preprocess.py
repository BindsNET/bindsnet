import cv2

def gray_scale(im):
    '''
    Converts RGB image into grayscale.
    
    Inputs:
        
        | :code:`im` (:code:`numpy.array`): RGB image.
    
    Returns:
        
        | :code:`im` (:code:`numpy.array`): Grayscaled image
    '''
    return cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)


def crop(im, x1, x2, y1, y2):
    return im[x1:x2, y1:y2, :]


def binary_image(im):
    '''
    Converts input image into black and white (binary)
    
    Inputs:
        
        | :code:`im` (:code:`numpy.array`): Grayscaled image
    
    Returns:
        
        | :code:`im` (:code:`numpy.array`): Black and white image. 
    '''
    _, im = cv2.threshold(im, 0, 1, cv2.THRESH_BINARY)
    return im


def subsample(im, x, y):
    '''
    Scale the image to (x, y).
    
    Inputs:
        
        | :code:`im` (:code:`numpy.array`): Image to be rescaled.
        | :code:`x` (:code:`int`): Output value for im's x dimension.
        | :code:`y` (:code:`int`): Output value for im's y dimension.
    
    Returns:
        
        | :code:`im` (:code:`numpy.array`): Rescaled image. 
    '''
    return cv2.resize(im, (x, y))                         
