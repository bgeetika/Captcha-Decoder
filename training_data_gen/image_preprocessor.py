import numpy

class ImagePreprocessor(object):
  @staticmethod
  def GetProcessedImageShape(image_shape):
    assert len(image_shape) == 3 and image_shape[2] >= 3, image_shape
    return (3, image_shape[0], image_shape[1])

  @staticmethod
  def ProcessImage(numpy_arr):
    assert len(numpy_arr.shape) == 3, numpy_arr.shape
    return numpy.rollaxis(numpy_arr, 2, 0)[:3, :, :]
