import numpy
import os
import re
import sys

from training_data_gen.image_preprocessor import ImagePreprocessor
import training_data_gen.vocabulary as vocabulary
import training_data_gen.utils as utils

CAPTCHA_FILENAME_PATTERN = re.compile('^\d+_(.*)\..+$')
def _ParseCaptchaFromImageFilename(image_filepath):
  image_filename = os.path.basename(image_filepath)
  match = CAPTCHA_FILENAME_PATTERN.match(image_filename)
  assert match is not None, image_filename
  return match.group(1)

def _GetCaptchaIdsFromImageFilename(image_filepath):
  captcha_str = _ParseCaptchaFromImageFilename(image_filepath)
  captcha_ids = numpy.zeros(len(captcha_str), dtype=numpy.int32)
  for i, captcha_char in enumerate(captcha_str):
    captcha_ids[i] = vocabulary.CHAR_VOCABULARY[captcha_char]
  return captcha_ids

def _GetShapeOfImagesUnderDir(captchas_dir):
  for captcha_filepath in utils.GetFilePathsUnderDir(captchas_dir):
    image_data = ImagePreprocessor.GetImageData(captcha_filepath)
    return image_data.shape
  return None

class TrainingData(object):
  @classmethod
  def Save(cls, file_path, image_data, chars):
    numpy.savez(file_path, image_data=image_data, chars=chars)

  @classmethod
  def Load(cls, file_path):
    training_data = numpy.load(file_path)
    return (ImagePreprocessor.NormalizeImageInput(training_data['image_data']),
            training_data['chars'])

  @classmethod
  def GenerateTrainingData(cls,
			   captchas_dir,
			   training_data_dir,
			   max_size=20000,
			   max_captcha_length=8):
    image_shape = _GetShapeOfImagesUnderDir(captchas_dir)
    training_data_shape = tuple(
	[max_size] + list(ImagePreprocessor.GetProcessedImageShape(image_shape)))
    training_image_data = numpy.zeros(training_data_shape, dtype=numpy.float32)
    training_labels = numpy.zeros((max_size, max_captcha_length),
                                  dtype=numpy.int32)

    i = 0
    for captcha_filepath in utils.GetFilePathsUnderDir(captchas_dir):
      try:
        image_data = ImagePreprocessor.GetImageData(captcha_filepath)
      except Exception as e:
        print e, captcha_filepath
        continue

      i += 1
      index = i % max_size
      training_image_data[index] = ImagePreprocessor.ProcessImage(image_data)
      captcha_ids = _GetCaptchaIdsFromImageFilename(captcha_filepath)
      training_labels[index, :] = numpy.zeros(max_captcha_length,
                                              dtype=numpy.int32)
      training_labels[index, :captcha_ids.shape[0]] = captcha_ids
   
      if i != 0 and (i % 10000) == 0:
        print 'Generated {0} examples.'.format(i)

      if i != 0 and i % max_size == 0:
        file_path = os.path.join(
            training_data_dir, "training_images_{0}.npy".format(i / max_size))
        try:
          cls.Save(file_path, training_image_data, training_labels)
        except Exception as e:
          print e


def main():
  TrainingData.GenerateTrainingData(sys.argv[1], sys.argv[2])

if __name__ == '__main__':
  main()
