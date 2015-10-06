import glob
import numpy
import os
import sys
import time

from model.nn_model import Model
import training_data_gen.utils as utils
from training_data_gen.image_preprocessor import ImagePreprocessor
from training_data_gen.training_data_generator import TrainingData
import training_data_gen.vocabulary as vocabulary

def IterateMinibatches(inputs, targets, batch_size, shuffle=False):
  assert len(inputs) == len(targets)
  if shuffle:
    indices = numpy.arange(len(inputs))
    numpy.random.shuffle(indices)
  for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
    if shuffle:
      excerpt = indices[start_idx:start_idx + batch_size]
    else:
      excerpt = slice(start_idx, start_idx + batch_size)
    yield inputs[excerpt], targets[excerpt]


BATCH_SIZE = 5000


def Train(train_fn, image_input, target_chars, batch_size=BATCH_SIZE):
  train_err = 0
  train_batches = 0
  start_time = time.time()
  print("Training")
  for image_input_batch, target_chars_batch in IterateMinibatches(
      image_input, target_chars, batch_size, shuffle=False):
    batch_start_time = time.time()
    train_err += train_fn(image_input_batch, target_chars_batch)
    train_batches += 1
    print("Batch training took {:.3f}s".format(time.time() - batch_start_time))
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))

  # Then we print the results for this epoch:
  print("Training took {:.3f}s".format(time.time() - start_time))
  print("  training loss:\t\t{:.6f}".format(train_err / train_batches))


def Test(test_fn, image_input, target_chars, batch_size=BATCH_SIZE):
  test_err = 0
  test_acc = 0
  test_batches = 0
  start_time = time.time()
  for image_input_batch, target_chars_batch in IterateMinibatches(
      image_input, target_chars, batch_size, shuffle=False):
    err, acc = test_fn(image_input_batch, target_chars_batch)
    test_err += err
    test_acc += acc
    test_batches += 1

  print("Testing took {:.3f}s".format(time.time() - start_time))
  print("  loss:\t\t{:.6f}".format(test_err / test_batches))
  print("  accuracy:\t\t{:.2f} %".format((test_acc / test_batches) * 100))

def _SaveModelWithPrefix(captcha_model, prefix):
  time_format = time.strftime('%Y_%m_%d_%H_%M_%S')
  def GetParamsFileName(file_prefix):
    return '{0}_{1}.npy.npz'.format(file_prefix, time_format)

  dir_path, file_prefix = os.path.dirname(prefix), os.path.basename(prefix)
  temp_params_file_prefix = os.path.join(dir_path, 'tmp.')
  temp_params_file = GetParamsFileName(temp_params_file_prefix)
  print 'Saving at temp location {0}'.format(temp_params_file)
  captcha_model.SaveModelParamsToFile(temp_params_file)

  perm_params_file = GetParamsFileName(prefix)
  print 'Saving at perm location {0}'.format(perm_params_file)
  os.rename(temp_params_file, perm_params_file)

def _KeepOnlyNModels(prefix, n=5):
  model_files = sorted(glob.glob('{0}*'.format(prefix)))
  models_to_delete = model_files[:-n]
  for model_file in models_to_delete:
    os.remove(model_file)

def _SaveModelAndRemoveOldOnes(captcha_model, prefix, n=5):
  _SaveModelWithPrefix(captcha_model, prefix)
  _KeepOnlyNModels(prefix, n)


def _GetLatestModelFile(params_file_prefix):
  param_files = glob.glob('{0}*'.format(params_file_prefix))
  if param_files:
    return max(glob.glob('{0}*'.format(params_file_prefix)))
  else:
    return None


def Run(training_data_dir, val_data_file, test_data_file,
        model_params_file_prefix, num_epochs=20):
  print('Compiling model')
  captcha_model = Model(
      saved_params_path=_GetLatestModelFile(model_params_file_prefix))
  print('Loading validation data')
  val_image_input, val_target_chars = TrainingData.Load(val_data_file)
  val_image_input = val_image_input[:BATCH_SIZE]
  val_target_chars = val_target_chars[:BATCH_SIZE]
  print('Starting training')
  for epoch_num in range(num_epochs):
    for i, training_file in enumerate(
        utils.GetFilePathsUnderDir(training_data_dir, shuffle=False)):
      image_input, target_chars = TrainingData.Load(training_file)
      Train(captcha_model.GetTrainFn(), image_input, target_chars)

      _SaveModelAndRemoveOldOnes(captcha_model, model_params_file_prefix)
      Test(captcha_model.GetTestFn(), image_input[:BATCH_SIZE],
           target_chars[:BATCH_SIZE])
      Test(captcha_model.GetTestFn(), val_image_input,
           val_target_chars)
      if i != 0 and i % 10 == 0:
        print 'Processed epoch:{0} {1} training files.'.format(epoch_num, i)

  test_image_input, test_target_chars = TrainingData.Load(test_data_file)
  Test(captcha_model.GetTestFn(), test_image_input, test_target_chars)


class CaptchaCracker(object):
  def __init__(self, model_params_file_prefix):
    latest_model_params_file = _GetLatestModelFile(model_params_file_prefix)
    captcha_model = Model(saved_params_path=latest_model_params_file)
    self._inference_fn = captcha_model.GetInferenceFn()

  def InferFromImagePath(self, image_path):
    image_input = ImagePreprocessor.ProcessImage(ImagePreprocessor.GetImageData(image_path))
    return self.InferForImageArray(image_input)

  def InferForImageArray(self, image_numpy_arr):
    image_numpy_arr = ImagePreprocessor.NormalizeImageInput(image_numpy_arr)
    predicted_char_id, predicted_probs = self._inference_fn(image_numpy_arr)
    char_vocabulary = vocabulary.CHAR_VOCABULARY
    predicted_char = char_vocabulary[predicted_char_id[0]]
    probs_by_chars = {}
    for i in range(predicted_probs.shape[1]):
      probs_by_chars[char_vocabulary[i]] = predicted_probs[0, i]
    return predicted_char, probs_by_chars


if __name__ == '__main__':
  Run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
