import numpy
import os
import re
import sys
from PIL import Image

from training_data_gen.image_preprocessor import ImagePreprocessor
import training_data_gen.vocabulary

CAPTCHA_FILENAME_PATTERN = re.compile('^\d+_(.*)\..+$')
def _ParseCaptchaFromImageFilename(image_filepath):
    image_filename = os.path.basename(image_filepath)
    match = CAPTCHA_FILENAME_PATTERN.match(image_filename)
    assert match is not None, image_filename
    return match.group(1)

def _GetCaptchaIdsFromImageFilename(image_filepath):
    captcha_str = _ParseCaptchaFromImageFilename(image_filepath)
    captcha_ids = numpy.zeros(len(captcha_str))
    for i, captcha_char in enumerate(captcha_str):
        captcha_ids[i] = vocabulary.CHAR_VOCABULARY[captcha_char]
    return captcha_ids


def _GetImageFilePathsUnderDir(captchas_dir):
  for image_filename in os.listdir(captchas_dir):
    captcha_filepath = os.path.join(captchas_dir, image_filename)
    if os.path.isfile(captcha_filepath):
      yield captcha_filepath

def _GetShapeOfImagesUnderDir(captchas_dir):
  for captcha_filepath in _GetImageFilePathsUnderDir(captchas_dir):
    image_data = numpy.asarray(Image.open(captcha_filepath)) 
    return image_data.shape
  return None

class TrainingData(object):
  @staticmethod
  def SaveTrainingData(file_path, image_data, chars):
    numpy.savez(file_path, image_data=image_data, chars=chars)

  @staticmethod
  def LoadTrainingData(file_path):
    training_data = numpy.load(file_path)
    return training_data['image_data'], training_data['chars']

  @staticmethod
  def GenerateTrainingData(captchas_dir,
			   training_data_dir,
			   max_size=50000,
			   max_captcha_length=5):
    image_shape = _GetShapeOfImagesUnderDir(captchas_dir)
    training_data_shape = tuple(
	[max_size] + list(ImagePreprocessor.GetProcessedImageShape(image_shape)))
    training_image_data = numpy.zeros(training_data_shape)
    training_labels = numpy.zeros((max_size, max_captcha_length))

    for i, captcha_filepath in enumerate(_GetImageFilePathsUnderDir(captchas_dir)):
      index = i % max_size
      image_data = numpy.asarray(Image.open(captcha_filepath))
      training_image_data[index] = ImagePreprocessor.ProcessImage(image_data)
      captcha_ids = _GetCaptchaIdsFromImageFilename(captcha_filepath)
      training_labels[index, :] = numpy.zeros(max_captcha_length)
      training_labels[index, :captcha_ids.shape[0]] = captcha_ids
      
      if i != 0 and i % max_size == 0:
	file_path = os.path.join(
	    training_data_dir, "training_images_{0}.npy".format(i / max_size))
        SaveTrainingData(file_path, training_image_data)


def main():
  TrainingData.GenerateTrainingData(sys.argv[1], sys.argv[2])

if __name__ == '__main__':
  main()
