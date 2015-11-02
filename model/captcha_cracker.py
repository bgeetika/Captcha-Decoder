from __future__ import print_function

import glob
import numpy
import os
import sys
import time
import parse_args

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


BATCH_SIZE = 500
TEST_BATCH_SIZE = 500


def Train(train_fn, image_input, target_chars, count_imagestrained_sofar, eval_matrix,  batch_size=BATCH_SIZE):
  train_err = 0
  train_batches = 0
  start_time = time.time()
  print("Training")
  for image_input_batch, target_chars_batch in IterateMinibatches(
      image_input, target_chars, batch_size, shuffle=False):
    batch_start_time = time.time()
    l_err = train_fn(image_input_batch, target_chars_batch)
    train_err += l_err
    train_batches += 1
    total_images_trained = count_imagestrained_sofar +  train_batches*BATCH_SIZE
    batch_training_time = time.time() - batch_start_time
    batch_training_loss = train_err / train_batches
    print("trained  {0}  images".format(total_images_trained))
    print("Batch training took {:.3f}s".format(batch_training_time))
    print("  training loss:\t\t{:.6f}".format(batch_training_loss))
    eval_matrix.num_of_images.append(total_images_trained)
    eval_matrix.training_time.append(batch_training_time)
    eval_matrix.training_loss.append(batch_training_loss)

  # Then we print the results for this epoch:
  print("Training took {:.3f}s".format(time.time() - start_time))
  print("  training loss:\t\t{:.6f}".format(train_err / train_batches))



def Test(test_fn, image_input, target_chars,
         multi_chars=True, batch_size=BATCH_SIZE):
  test_err = 0
  test_acc = 0
  seq_test_acc = 0
  test_batches = 0
  start_time = time.time()
  for image_input_batch, target_chars_batch in IterateMinibatches(
      image_input, target_chars, batch_size, shuffle=False):
    if multi_chars:
      err, acc, seq_acc = test_fn(image_input_batch, target_chars_batch)
    else:
      err, acc = test_fn(image_input_batch, target_chars_batch)
    test_err += err
    test_acc += acc
    seq_test_acc += seq_acc
    test_batches += 1

  print("Testing took {:.3f}s".format(time.time() - start_time))
  print("  loss:\t\t{:.6f}".format(test_err / test_batches))
  print("  char accuracy:\t\t{:.2f} %".format((test_acc / test_batches) * 100))
  print("  seq accuracy:\t\t{:.2f} %".format((seq_test_acc / test_batches) * 100))

def _SaveModelWithPrefix(captcha_model, prefix):
  time_format = time.strftime('%Y_%m_%d_%H_%M_%S')
  def GetParamsFileName(file_prefix):
    return '{0}_{1}.npy.npz'.format(file_prefix, time_format)

  dir_path, file_prefix = os.path.dirname(prefix), os.path.basename(prefix)
  temp_params_file_prefix = os.path.join(dir_path, 'tmp.')
  temp_params_file = GetParamsFileName(temp_params_file_prefix)
  print('Saving at temp location {0}'.format(temp_params_file))
  captcha_model.SaveModelParamsToFile(temp_params_file)

  perm_params_file = GetParamsFileName(prefix)
  print('Saving at perm location {0}'.format(perm_params_file))
  os.rename(temp_params_file, perm_params_file)

def _KeepOnlyNModels(prefix, n=5):
  model_files = sorted(glob.glob('{0}*'.format(prefix)))
  models_to_delete = model_files[:-n]
  for model_file in models_to_delete:
    os.remove(model_file)

def _SaveModelAndRemoveOldOnes(captcha_model, prefix, n=5):
  _SaveModelWithPrefix(captcha_model, prefix)
  _KeepOnlyNModels(prefix, n)


def GetLatestModelFile(params_file_prefix):
  param_files = glob.glob('{0}*'.format(params_file_prefix))
  if param_files:
    return max(glob.glob('{0}*'.format(params_file_prefix)))
  else:
    return None

class EvalMatrix:
     def __init__(self):
         self.num_of_images = []
         self.training_time = []
         self.training_acc = []
         self.training_loss = []

def Run(args, num_epochs=20, multi_chars=True, num_softmaxes=None):
  training_data_dir = args.TrainingDirc
  val_data_file = args.ValidateDirc
  test_data_file = args.TestDirc
  model_params_file_prefix = args.ModelParamsFile
  global BATCH_SIZE
  global TEST_BATCH_SIZE
  includeCapital = args.includeCapital
  length = 5 if not args.length else int(args.length)
           
  if args.maxsoft:
       num_softmaxes = int(args.maxsoft)
  if args.batchsize:
       BATCH_SIZE = int(args.batchsize)
  if args.testsize:
       TEST_BATCH_SIZE = int(args.testsize)
  learning_rate = args.learningrate
  no_hidden_layers = args.hiddenlayers

  print('Compiling model')
  captcha_model = Model(learning_rate, no_hidden_layers,includeCapital,length, 
      saved_params_path=GetLatestModelFile(model_params_file_prefix),
      multi_chars=multi_chars, num_softmaxes=num_softmaxes)
  print('Loading validation data')
  val_image_input, val_target_chars = TrainingData.Load(val_data_file)
  val_image_input = val_image_input[:TEST_BATCH_SIZE]
  val_target_chars = val_target_chars[:TEST_BATCH_SIZE]
  eval_matrix = EvalMatrix()
  print('Starting training')
  total_images_trained = 0
  #_SaveModelAndRemoveOldOnes(captcha_model, model_params_file_prefix)
  for epoch_num in range(num_epochs):
    for i, training_file in enumerate(
        utils.GetFilePathsUnderDir(training_data_dir, shuffle=False)):
      image_input, target_chars = TrainingData.Load(training_file)
      Train(captcha_model.GetTrainFn(), image_input, target_chars,
            total_images_trained, eval_matrix, BATCH_SIZE)
      total_images_trained += image_input.shape[0]
      _SaveModelAndRemoveOldOnes(captcha_model, model_params_file_prefix)
      Test(captcha_model.GetTestFn(), image_input[:TEST_BATCH_SIZE],
           target_chars[:TEST_BATCH_SIZE],batch_size=BATCH_SIZE, multi_chars=multi_chars)
      Test(captcha_model.GetTestFn(), val_image_input,
           val_target_chars, batch_size=BATCH_SIZE, multi_chars=multi_chars)
      if i != 0 and i % 10 == 0:
        print('Processed epoch:{0} {1} training files.'.format(epoch_num + 1 , i))
  
  
  test_image_input, test_target_chars = TrainingData.Load(test_data_file)
  Test(captcha_model.GetTestFn(), test_image_input,
       test_target_chars, multi_chars=multi_chars)
  num_of_images_vector = np.narray(eval_matrix.num_of_images)
  training_time_vector = np.narray(eval_matrix.training_time)
  training_loss_vector = np.narray(eval_matrix.training_loss)  
 
class CaptchaCracker(object):
  def __init__(self, model_params_file_prefix, graph_file_path= "/home/geetika/model_graph.png",  multi_chars=True, num_softmaxes=None):
    latest_model_params_file = GetLatestModelFile(model_params_file_prefix)
    self.captcha_model = Model(saved_params_path=latest_model_params_file, includeCapital=True, multi_chars=multi_chars, num_softmaxes=num_softmaxes)
    self.captcha_model.GetPrettyPrint(graph_file_path)
    self._inference_fn = self.captcha_model.GetInferenceFn()

  def InferFromImagePath(self, image_path):
    image_input = ImagePreprocessor.ProcessImage(ImagePreprocessor.GetImageData(image_path))
    return self.InferForImageArray(image_input)

  def InferForImageArray(self, image_numpy_arr):
    image_numpy_arr = ImagePreprocessor.NormalizeImageInput(image_numpy_arr)
    predicted_char_id, predicted_probs = self._inference_fn(
        numpy.expand_dims(image_numpy_arr, axis=0))
    chars = self.captcha_model.CHARS
    if predicted_char_id.ndim == 1:
      predicted_chars = chars[predicted_char_id[0]]
      probs_by_chars = {}
      for i in range(predicted_probs.shape[1]):
	probs_by_chars[chars[i]] = predicted_probs[0, i]
    else:
      assert predicted_char_id.ndim == 2, predicted_char_id.shape
      predicted_chars = [0] * predicted_char_id.shape[1]
      probs_by_chars = [{} for _ in range(predicted_char_id.shape[1])]
      for i in range(predicted_char_id.shape[1]):
        predicted_chars[i] = chars[predicted_char_id[0, i]]
	for j in range(predicted_probs.shape[2]):
	  probs_by_chars[i][chars[j]] = predicted_probs[0, i, j]
    return predicted_chars, probs_by_chars


if __name__ == '__main__':
  arguments = parse_args.parse_arg()
  Run(arguments)
