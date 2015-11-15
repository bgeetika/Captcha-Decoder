import numpy
from PIL import Image

class ImagePreprocessor(object):
  @classmethod
  def GetProcessedImageShape(cls, image_shape):
    #assert len(image_shape) == 3 and image_shape[2] >= 1, image_shape
    #return (1, image_shape[0], image_shape[1])
    assert len(image_shape) == 2, image_shape
    return (1, image_shape[0], image_shape[1])

  @classmethod
  def ProcessImage(cls, numpy_arr):
    #assert len(numpy_arr.shape) == 3, numpy_arr.shape
    #return numpy.rollaxis(numpy_arr, 2, 0)[:1, :, :]
    assert len(numpy_arr.shape) == 2, numpy_arr.shape
    return numpy.expand_dims(numpy_arr, axis=0)

  @classmethod
  def GetImageData(cls, image_file_path):
    return numpy.asarray(Image.open(image_file_path).convert('L'))

  @classmethod
  def NormalizeImageInput(cls, image_input):
    return image_input / 100.0

  @classmethod
  def RescaleImageInput(cls, image_input):
    image_input[image_input > numpy.percentile(image_input, 88)] = numpy.max(image_input)
    image_input = image_input / numpy.max(image_input)
    return image_input
