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
import csv

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

def GetMaskInput(target_chars):
  mask_input = target_chars.copy()
  mask_input[:, 0] = 1
  mask_input[numpy.roll(mask_input, 1, axis=1) > 0] = 1
  return mask_input

def Train(train_fn, image_input, target_chars, count_imagestrained_sofar, eval_matrix,
          batch_size=BATCH_SIZE, use_mask_input=False):
  train_err = 0
  train_batches = 0
  start_time = time.time()
  print("Training")
  total_images_trained = 0
  for image_input_batch, target_chars_batch in IterateMinibatches(
      image_input, target_chars, batch_size, shuffle=False):
    batch_start_time = time.time()
    train_inputs = [image_input_batch, target_chars_batch]
    if use_mask_input:
      train_inputs.append(GetMaskInput(target_chars_batch))
    l_err = train_fn(*train_inputs)
    train_err += l_err
    train_batches += 1
    total_images_trained = count_imagestrained_sofar +  train_batches*BATCH_SIZE
    batch_training_time = time.time() - batch_start_time
    batch_training_loss = train_err / train_batches
    print("trained  {0}  images".format(total_images_trained))
    print("Batch training took {:.3f}s".format(batch_training_time))
    print("  training loss:\t\t{:.6f}".format(batch_training_loss))
 
  training_time = (time.time() - start_time)
  training_loss = (train_err / train_batches)
  print (training_loss , type(training_loss))
  # Then we print the results for this epoch:
  print("Training took {:.3f}s".format(training_time))
  eval_matrix.num_of_images_training_time.append((total_images_trained, training_time))
  print("training loss:\t\t{:.6f}".format(training_loss))
  eval_matrix.num_of_images_training_loss.append((total_images_trained,training_loss))



def Test(test_fn, image_input, target_chars, total_images_trained, train_flag, eval_matrix,
         multi_chars=True, batch_size=BATCH_SIZE, use_mask_input=False):
  test_err = 0
  test_acc = 0
  seq_test_acc = 0
  test_batches = 0
  start_time = time.time()
  
  for image_input_batch, target_chars_batch in IterateMinibatches(
      image_input, target_chars, batch_size, shuffle=False):
    test_inputs = [image_input_batch, target_chars_batch]
    if use_mask_input:
      test_inputs.append(GetMaskInput(target_chars_batch))
    out = test_fn(*test_inputs)
    if multi_chars:
      err, acc, seq_acc = tuple(out)
    else:
      err, acc = tuple(out)
    test_err += err
    test_acc += acc
    seq_test_acc += seq_acc
    test_batches += 1
  char_acc = ((test_acc / test_batches) * 100)
  seq_acc = ((seq_test_acc / test_batches) * 100)
  if train_flag:
        print("Testing on training took {:.3f}s".format(time.time() - start_time))
        eval_matrix.number_of_images_training_acc.append((total_images_trained, char_acc, seq_acc)) 
  else:      
        print("Testing on testing data took {:.3f}s".format(time.time() - start_time))
        eval_matrix.number_of_images_testing_acc.append((total_images_trained, char_acc, seq_acc))
        
  print("  loss:\t\t{:.6f}".format(test_err / test_batches))
  print("  char accuracy:\t\t{:.2f} %".format(char_acc))
  print("  seq accuracy:\t\t{:.2f} %".format(seq_acc))

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
     def __init__(self, prefix):
         parent_direc = os.path.dirname(prefix)
         print (prefix)
         prefix = prefix.rsplit("/", 1)[1]
         print (prefix)
         self.training_time = os.path.join(parent_direc,  "_training_time_" + prefix + ".csv")
         self.training_loss = os.path.join(parent_direc,  "_training_loss_"+prefix +".csv")
         self.training_acc =  os.path.join(parent_direc,   "_training_acc_" + prefix + ".csv")
         self.testing_acc  =  os.path.join(parent_direc,  "_testing_acc_"+ prefix + ".csv")
         self.num_of_images_training_time = []
         self.num_of_images_training_loss = []
         self.number_of_images_training_acc = []
         self.number_of_images_testing_acc = []
     
     def update_files(self):
        with open(self.training_time, 'a') as csvfile:
          csvwriter = csv.writer(csvfile)
          for elem in self.num_of_images_training_time:
              csvwriter.writerow(elem)
        
        with open(self.training_loss, 'a') as csvfile:
          csvwriter = csv.writer(csvfile)
          for elem in self.num_of_images_training_loss:
              csvwriter.writerow(elem)
        
        with open(self.training_acc, 'a') as csvfile:
          csvwriter = csv.writer(csvfile)
          for elem in self.number_of_images_training_acc:
              csvwriter.writerow(elem)

        with open(self.testing_acc, 'a') as csvfile:
          csvwriter = csv.writer(csvfile)
          for elem in self.number_of_images_testing_acc:
              csvwriter.writerow(elem)

def Run(args, num_epochs=20, multi_chars=True, num_softmaxes=None):
  training_data_dir = args.TrainingDirc
  val_data_file = args.ValidateDirc
  test_data_file = args.TestDirc
  multi_char = args.multichar
  model_params_file_prefix = args.ModelParamsFile
  global BATCH_SIZE
  global TEST_BATCH_SIZE
  includeCapital = args.includeCapital
  length = 5 if not args.length else int(args.length)
  bidirec = args.bidirec 
  cnn_dense_layer_sizes = [int(args.cnn_dense_layer_sizes)]
  print (cnn_dense_layer_sizes, bidirec)
  if args.maxsoft:
       num_softmaxes = int(args.maxsoft)
  if args.batchsize:
       BATCH_SIZE = int(args.batchsize)
  if args.testsize:
       TEST_BATCH_SIZE = int(args.testsize)
  learning_rate = args.learningrate
  lstm_grad_clipping = args.lstm_grad_clipping
  if lstm_grad_clipping == None:
        lstm_grad_clipping = False
  elif lstm_grad_clipping.upper() == "True".upper():
        lstm_grad_clipping = True
  elif lstm_grad_clipping.upper() == "False".upper():
        lstm_grad_clipping = False
  else:
      lstm_grad_clipping = float(lstm_grad_clipping)
  no_hidden_layers = args.hiddenlayers
  lstm_layer_units=args.lstm_layer_units
  print('Compiling model')
  captcha_model = Model(
      learning_rate=learning_rate,
      no_hidden_layers=no_hidden_layers,
      includeCapital=includeCapital,
      num_rnn_steps=length, 
      saved_params_path=GetLatestModelFile(model_params_file_prefix),
      multi_chars=multi_chars, num_softmaxes=num_softmaxes,
      use_mask_input=args.use_mask_input,
      lstm_layer_units=lstm_layer_units,
      cnn_dense_layer_sizes = cnn_dense_layer_sizes,
      bidirec = bidirec,
      lstm_grad_clipping=lstm_grad_clipping)
  print('Loading validation data')
  val_image_input, val_target_chars = TrainingData.Load(val_data_file, rescale_in_preprocessing=args.rescale)
  val_image_input = val_image_input[:TEST_BATCH_SIZE]
  val_target_chars = val_target_chars[:TEST_BATCH_SIZE]
  eval_matrix = EvalMatrix(model_params_file_prefix)
  print('Starting training')
  total_images_trained = 0
  _SaveModelAndRemoveOldOnes(captcha_model, model_params_file_prefix)
  for epoch_num in range(num_epochs):
    for i, training_file in enumerate(
        utils.GetFilePathsUnderDir(training_data_dir, shuffle=True)):
      image_input, target_chars = TrainingData.Load(training_file, rescale_in_preprocessing=args.rescale)
      Train(captcha_model.GetTrainFn(), image_input, target_chars,
            total_images_trained, eval_matrix, BATCH_SIZE,
            use_mask_input=args.use_mask_input)
      _SaveModelAndRemoveOldOnes(captcha_model, model_params_file_prefix)
      total_images_trained += image_input.shape[0]
      train_flag = True
      test_flag = False
      Test(captcha_model.GetTestFn(), image_input[:TEST_BATCH_SIZE],
           target_chars[:TEST_BATCH_SIZE],total_images_trained, train_flag, eval_matrix,
           batch_size=TEST_BATCH_SIZE, multi_chars=multi_chars,
           use_mask_input=args.use_mask_input)
      Test(captcha_model.GetTestFn(), val_image_input,
           val_target_chars, total_images_trained, test_flag, eval_matrix,
           batch_size=TEST_BATCH_SIZE, multi_chars=multi_chars,
           use_mask_input=args.use_mask_input)
      eval_matrix.update_files()
      eval_matrix = EvalMatrix(model_params_file_prefix)
      if i != 0 and i % 10 == 0:
        print('Processed epoch:{0} {1} training files.'.format(epoch_num + 1 , i))
  
  
  test_image_input, test_target_chars = TrainingData.Load(
      test_data_file, rescale_in_preprocessing=args.rescale)
  Test(captcha_model.GetTestFn(), test_image_input,
       test_target_chars, multi_chars=multi_chars,
       use_mask_input=args.use_mask_input)
 
class CaptchaCracker(object):
  def __init__(self, model_params_file_prefix, includeCapital,
               graph_file_path= "/home/geetika/model_graph.png",
               multi_chars=True, num_softmaxes=None, rescale_in_preprocessing=False,
               num_rnn_steps=5, use_mask_input=False, lstm_layer_units = 256, cnn_dense_layer_sizes = [256],
               lstm_grad_clipping = False, bidirec = False):
    latest_model_params_file = GetLatestModelFile(model_params_file_prefix)
    if type(cnn_dense_layer_sizes) == type(1):
           cnn_dense_layer_sizes = [cnn_dense_layer_sizes]
    self.captcha_model = Model(
        saved_params_path=latest_model_params_file, includeCapital=includeCapital,
        multi_chars=multi_chars, num_softmaxes=num_softmaxes, num_rnn_steps=num_rnn_steps,
        use_mask_input=use_mask_input, lstm_layer_units=lstm_layer_units, cnn_dense_layer_sizes = cnn_dense_layer_sizes,
        lstm_grad_clipping = lstm_grad_clipping, bidirec=bidirec )
    self._inference_fn = self.captcha_model.GetInferenceFn()
    self._rescale_in_preprocessing = rescale_in_preprocessing
    self._num_rnn_steps = num_rnn_steps
    self._use_mask_input = use_mask_input

  def InferFromImagePath(self, image_path):
    image_input = ImagePreprocessor.ProcessImage(ImagePreprocessor.GetImageData(image_path))
    return self.InferForImageArray(image_input.copy())

  def InferForImageArray(self, image_numpy_arr):
    if self._rescale_in_preprocessing:
      image_numpy_arr = ImagePreprocessor.RescaleImageInput(image_numpy_arr)
    else:
      image_numpy_arr = ImagePreprocessor.NormalizeImageInput(image_numpy_arr)
    inference_inputs = [numpy.expand_dims(image_numpy_arr, axis=0)]
    if self._use_mask_input:
      inference_inputs.append(numpy.ones((1, self._num_rnn_steps)))
    predicted_char_id, predicted_probs = self._inference_fn(*inference_inputs)
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
