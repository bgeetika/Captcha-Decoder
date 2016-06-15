from __future__ import print_function

import lasagne
import numpy
import theano
import theano.tensor as T

import training_data_gen.vocabulary as vocabulary

class Model(object):
  '''
  class for creating a model.
  '''
  def __init__(self, learning_rate=0.01, no_hidden_layers=2, includeCapital=False,
               num_rnn_steps=5,  saved_params_path=None, multi_chars=True,
               num_softmaxes=None, use_mask_input=False, lstm_layer_units=256):
    if not learning_rate:
           self.learning_rate = 0.01
    else:
           self.learning_rate = float(learning_rate)
    if not no_hidden_layers:
           self.no_hidden_layers=2
    else:
           self.no_hidden_layers = int(no_hidden_layers)
    self.num_rnn_steps = num_rnn_steps
    self.includeCapital = includeCapital
    self.CHAR_VOCABULARY, self.CHARS = vocabulary.GetCharacterVocabulary(includeCapital)
    if multi_chars:
      if num_softmaxes:
	self._network, self._train_fn, self._test_fn, self._inference_fn = (
	    self._InitializeModelThatPredictsCharsMultiSoftmax(self.learning_rate, num_softmaxes=num_softmaxes))
      else:
	self._network, self._train_fn, self._test_fn, self._inference_fn = (
	    self._InitializeModelThatPredictsAllChars(
                self.learning_rate, use_mask_input=use_mask_input,
                lstm_layer_units=lstm_layer_units))
    else:
      self._network, self._train_fn, self._test_fn, self._inference_fn = (
	  self._InitializeModelThatPredictsFirstChar(self.learning_rate))
    self.prediction = lasagne.layers.get_output(self._network) 
    if saved_params_path:
      ''' 
      If saved params path is specified, then start from that parms value.
      '''
      f = numpy.load(saved_params_path)
      param_values = [f['arr_%d' % i] for i in range(len(f.files))]
      '''
      Deleting a vector as it is not needed. It was taking memory. so needed to delete it explicitly.
      '''
      del f.f
      f.close()
      lasagne.layers.set_all_param_values(self._network, param_values)

  def SaveModelParamsToFile(self, file_path):
    '''
    It saved the current parms to a file for future use.
    '''
    numpy.savez(file_path, *lasagne.layers.get_all_param_values(self._network))

  def GetTrainFn(self):
    return self._train_fn

  def GetTestFn(self):
    return self._test_fn

  def GetPrettyPrint(self, file_path):
      theano.printing.pydotprint(self._inference_fn, outfile=file_path, var_with_name_simple=True)
      
  def GetInferenceFn(self):
    return self._inference_fn

  class CNNMaxPoolConfig(object):
    def __init__(self, num_cnn_filters, cnn_filter_size, max_pool_size):
      self.num_cnn_filters = num_cnn_filters
      self.cnn_filter_size = cnn_filter_size
      self.max_pool_size = max_pool_size

  @classmethod
  def _InitializeModelThatPredictsFirstChar(cls,learning_rate):
    image_input = T.tensor4('image_input')
    prediction_layer = cls._BuildModelToPredictFirstChar(image_input)

    target_chars = T.imatrix('target_chars')
    target_char = target_chars[:, 0]

    # Create a loss expression for training, Using cross-entropy loss.
    prediction = lasagne.layers.get_output(prediction_layer)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_char)
    loss = loss.mean()

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum.
    params = lasagne.layers.get_all_params(prediction_layer, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
     	loss, params, learning_rate, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(prediction_layer, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
							    target_char)
    test_loss = test_loss.mean()

    predicted_char = T.argmax(test_prediction, axis=1)
    # An expression for the classification accuracy:
    test_acc = T.mean(T.eq(predicted_char, target_char),
		      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([image_input, target_chars], loss, updates=updates,
                               allow_input_downcast=True)

    # Compile a second function computing the prediction, validation loss and accuracy:
    test_fn = theano.function([image_input, target_chars],
			      [test_loss, test_acc],
                              allow_input_downcast=True)

    # Compile a third function computing the prediction.
    inference_fn = theano.function([image_input],
			           [predicted_char, test_prediction],
                                   allow_input_downcast=True)

    return prediction_layer, train_fn, test_fn, inference_fn


  def _BuildModelToPredictFirstChar(
      self,
      image_input,
      cnn_max_pool_configs=None,
      cnn_dense_layer_sizes=[512]):
    if cnn_max_pool_configs is None:
      cnn_max_pool_configs = self._DefaultCNNMaxPoolConfigs()
    network = lasagne.layers.InputLayer(shape=(None, 1, 50, 200),
                                        input_var=image_input)
    network = self._BuildCNN(network, cnn_max_pool_configs, cnn_dense_layer_sizes)

    # And, finally, the softmax layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=len(self.CHARS),
            nonlinearity=lasagne.nonlinearities.softmax)
    return network

  def _InitializeModelThatPredictsCharsMultiSoftmax(self,learning_rate, num_softmaxes=5):
    image_input = T.tensor4('image_input')
    print ("num_of_softmax: " + str(num_softmaxes))
    #prediction_layer = self._BuildModelToPredictFirstChar(image_input)
    prediction_layer = self._BuildModelToPredictCharsMultiSoftmax(
        image_input, num_softmaxes=num_softmaxes)

    target_chars_input = T.imatrix('target_chars_input')
    target_chars = target_chars_input[:, :num_softmaxes].reshape(shape=(-1,))

    # Create a loss expression for training, Using cross-entropy loss.
    prediction = lasagne.layers.get_output(prediction_layer)
    l_loss = lasagne.objectives.categorical_crossentropy(prediction, target_chars)
    loss = l_loss.mean()

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum.
    params = lasagne.layers.get_all_params(prediction_layer, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
    	loss, params, learning_rate, momentum=0.9)
    #updates = lasagne.updates.adagrad(loss, params, learning_rate=0.0001)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(prediction_layer, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
							    target_chars)
    test_loss = test_loss.mean()

    predicted_chars = T.argmax(test_prediction, axis=1)
    correctly_predicted_chars = T.eq(predicted_chars, target_chars)
    # An expression for the classification accuracy:
    test_acc = T.mean(correctly_predicted_chars,
		      dtype=theano.config.floatX)
    predicted_chars = predicted_chars.reshape(shape=(-1, num_softmaxes))
    correctly_predicted_chars = correctly_predicted_chars.reshape(shape=(-1, num_softmaxes))
    num_chars_matched = T.sum(correctly_predicted_chars, axis=1, dtype=theano.config.floatX)
    seq_test_acc = T.mean(T.eq(num_chars_matched, T.fill(num_chars_matched, num_softmaxes)),
                          dtype=theano.config.floatX)
    test_prediction = test_prediction.reshape(shape=(-1, num_softmaxes, len(self.CHARS)))

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function(
        [image_input, target_chars_input],
        loss,
        updates=updates,
        allow_input_downcast=True)

    # Compile a second function computing the prediction, validation loss and accuracy:
    test_fn = theano.function([image_input, target_chars_input],
			      [test_loss, test_acc, seq_test_acc],
                              allow_input_downcast=True)

    # Compile a third function computing the prediction.
    inference_fn = theano.function([image_input],
			           [predicted_chars, test_prediction],
                                   allow_input_downcast=True)

    return prediction_layer, train_fn, test_fn, inference_fn


  def _BuildModelToPredictCharsMultiSoftmax(
      self,
      image_input,
      num_softmaxes=5,
      cnn_max_pool_configs=None,
      cnn_dense_layer_sizes=[256],
      softmax_dense_layer_size=256):
    if cnn_max_pool_configs is None:
      cnn_max_pool_configs = self._DefaultCNNMaxPoolConfigs()
    network = lasagne.layers.InputLayer(shape=(None, 1, 50, 200),
                                        input_var=image_input)
    cnn_dense_layer_sizes = [x*num_softmaxes for x in cnn_dense_layer_sizes]
    network = self._BuildCNN(network, cnn_max_pool_configs, cnn_dense_layer_sizes)
    #network = self._BuildImageNetCNN(network)

    l_dense_layers = []
    for _ in range(num_softmaxes):
      l_dense_layer = lasagne.layers.DenseLayer(
	      lasagne.layers.dropout(network, p=.5),
	      num_units=softmax_dense_layer_size,
	      nonlinearity=lasagne.nonlinearities.rectify)
      #l_dense_layer = lasagne.layers.DimshuffleLayer(l_dense_layer, (0, 'x', 1))
      l_dense_layer = lasagne.layers.ReshapeLayer(l_dense_layer, ([0], 1, [1]))
      l_dense_layers.append(l_dense_layer)

    l_dense = lasagne.layers.ConcatLayer(l_dense_layers, axis=1)
    l_dense = lasagne.layers.ReshapeLayer(l_dense, (-1, [2]))
    l_softmax = lasagne.layers.DenseLayer(
	    lasagne.layers.dropout(l_dense, p=.5),
	    num_units=len(self.CHARS),
	    nonlinearity=lasagne.nonlinearities.softmax)
    return l_softmax


  def _InitializeModelThatPredictsAllChars(
      self, learning_rate, bidirectional_rnn=False, use_mask_input=False,
      lstm_layer_units=256):
    image_input = T.tensor4('image_input')
    num_rnn_steps = self.num_rnn_steps
    target_chars_input = T.imatrix('target_chars')
    target_chars = target_chars_input[:, :num_rnn_steps]
    target_chars = target_chars.reshape(shape=(-1,))

    mask_input_input = None
    mask_input = None
    if use_mask_input:
      mask_input_input = T.imatrix('mask_input')
      mask_input = mask_input_input[:, :num_rnn_steps]
      #mask_input = mask_input.reshape(shape=(-1,))
    prediction_layer, l_cnn, l_lstm = self._BuildModelToPredictAllChars(
        image_input, num_rnn_steps=num_rnn_steps, mask_input=mask_input,
        bidirectional_rnn=bidirectional_rnn, lstm_layer_units=lstm_layer_units)

    # Create a loss expression for training, Using cross-entropy loss.
    #prediction = lasagne.layers.get_output(prediction_layer)
    prediction, l_cnn, l_lstm = tuple(
        lasagne.layers.get_output([prediction_layer, l_cnn, l_lstm]))
    l_loss = lasagne.objectives.categorical_crossentropy(prediction, target_chars)
    if use_mask_input:
      l_loss = l_loss.reshape(shape=(-1, num_rnn_steps))
      l_loss *= mask_input
      loss = l_loss.sum() / mask_input.sum()
    else:
      loss = l_loss.mean()

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum.
    params = lasagne.layers.get_all_params(prediction_layer, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
     	loss, params, learning_rate, momentum=0.9)
    #updates = lasagne.updates.adagrad(loss, params, learning_rate=0.001)

    grads = theano.grad(loss, params)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(prediction_layer, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
							    target_chars)
    test_loss = test_loss.mean()

    predicted_chars = T.argmax(test_prediction, axis=1)
    correctly_predicted_chars = T.eq(predicted_chars, target_chars)
    # An expression for the classification accuracy:
    test_acc = T.mean(correctly_predicted_chars,
		      dtype=theano.config.floatX)
    predicted_chars = predicted_chars.reshape(shape=(-1, num_rnn_steps))
    correctly_predicted_chars = correctly_predicted_chars.reshape(shape=(-1, num_rnn_steps))
    num_chars_matched = T.sum(correctly_predicted_chars, axis=1, dtype=theano.config.floatX)
    seq_test_acc = T.mean(T.eq(num_chars_matched, T.fill(num_chars_matched, num_rnn_steps)),
                          dtype=theano.config.floatX)
    test_prediction = test_prediction.reshape(shape=(-1, num_rnn_steps, len(self.CHARS)))

    mask_input_vec = [mask_input_input] if use_mask_input else []
    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function(
        [image_input, target_chars_input] + mask_input_vec,
        loss,
        updates=updates,
        allow_input_downcast=True)

    # Compile a second function computing the prediction, validation loss and accuracy:
    test_fn = theano.function([image_input, target_chars_input] + mask_input_vec,
			      [test_loss, test_acc, seq_test_acc],
                              allow_input_downcast=True)

    # Compile a third function computing the prediction.
    inference_fn = theano.function([image_input] + mask_input_vec,
			           [predicted_chars, test_prediction],
                                   allow_input_downcast=True)

    return prediction_layer, train_fn, test_fn, inference_fn


  def _BuildModelToPredictAllChars(
      self,
      image_input,
      num_rnn_steps,
      mask_input=None,
      cnn_max_pool_configs=None,
      cnn_dense_layer_sizes=[256],
      lstm_layer_units=256,
      lstm_precompute_input=True,
      bidirectional_rnn=False,
      lstm_unroll_scan=False):
    if cnn_max_pool_configs is None:
      cnn_max_pool_configs = self._DefaultCNNMaxPoolConfigs()
    network = lasagne.layers.InputLayer(shape=(None, 1, 50, 200),
                                        input_var=image_input)
    if mask_input:
      mask_input = lasagne.layers.InputLayer(shape=(None, num_rnn_steps),
					     input_var=mask_input)
    l_cnn = self._BuildCNN(network, cnn_max_pool_configs, cnn_dense_layer_sizes)

    l_cnn = lasagne.layers.ReshapeLayer(l_cnn, ([0], 1, [1]))
    l_rnn_input = lasagne.layers.ConcatLayer([l_cnn for _ in range(num_rnn_steps)], axis=1)


    l_forward_lstm = lasagne.layers.LSTMLayer(
        l_rnn_input,
        num_units=lstm_layer_units,
        #forgetgate=lasagne.layers.Gate(b=lasagne.init.Constant(5.0)),
        mask_input=mask_input,
        precompute_input=lstm_precompute_input,
        unroll_scan=lstm_unroll_scan)
    l_lstm = None
    if bidirectional_rnn:
      l_backward_lstm = lasagne.layers.LSTMLayer(
	  l_rnn_input,
	  num_units=lstm_layer_units,
	  mask_input=mask_input,
	  precompute_input=lstm_precompute_input,
          unroll_scan=lstm_unroll_scan,
          backwards=True)
      l_lstm = lasagne.layers.ConcatLayer([l_forward_lstm, l_backward_lstm], axis=2)
      l_lstm = lasagne.layers.ReshapeLayer(l_lstm, (-1, 2*lstm_layer_units))
      l_lstm = lasagne.layers.DenseLayer(
	      lasagne.layers.dropout(l_lstm, p=.5),
	      num_units=lstm_layer_units)
    else:
      l_lstm = l_forward_lstm
      l_lstm = lasagne.layers.ReshapeLayer(l_lstm, (-1, lstm_layer_units))
    
    # And, finally, the softmax layer with 50% dropout on its inputs:
    l_softmax = lasagne.layers.DenseLayer(
	lasagne.layers.dropout(l_lstm, p=.5),
	num_units=len(self.CHARS),
	nonlinearity=lasagne.nonlinearities.softmax)
    return l_softmax, l_cnn, l_lstm


  @classmethod
  def _BuildCNN(cls,
               network,
               cnn_max_pool_configs,
	       cnn_dense_layer_sizes):
    for config in cnn_max_pool_configs:
      # Convolutional layer
      network = lasagne.layers.Conv2DLayer(
	  network, num_filters=config.num_cnn_filters, filter_size=config.cnn_filter_size,
	  nonlinearity=lasagne.nonlinearities.rectify,
	  W=lasagne.init.GlorotUniform())

      # Max-pooling layer.
      network = lasagne.layers.MaxPool2DLayer(network, pool_size=config.max_pool_size)

    for dense_layer_size in cnn_dense_layer_sizes:
      # A fully-connected layer with 50% dropout on its inputs:
      network = lasagne.layers.DenseLayer(
	      lasagne.layers.dropout(network, p=.5),
	      num_units=dense_layer_size,
	      nonlinearity=lasagne.nonlinearities.rectify)
    return network

  @classmethod
  def _BuildImageNetCNN(cls, in_layer):
    ConvLayer = lasagne.layers.Conv2DLayer
    DenseLayer = lasagne.layers.DenseLayer
    DropoutLayer = lasagne.layers.DropoutLayer
    PoolLayer = lasagne.layers.MaxPool2DLayer
    NormLayer = lasagne.layers.LocalResponseNormalization2DLayer

    l_layer = in_layer
    l_layer = ConvLayer(l_layer, num_filters=96, filter_size=7, stride=2)
    l_layer = NormLayer(l_layer, alpha=0.0001) # caffe has alpha = alpha * pool_size
    l_layer = PoolLayer(l_layer, pool_size=3, stride=3, ignore_border=False)
    l_layer = ConvLayer(l_layer, num_filters=256, filter_size=5)
    l_layer = PoolLayer(l_layer, pool_size=2, stride=2, ignore_border=False)
    l_layer = ConvLayer(l_layer, num_filters=512, filter_size=3, pad=1)
    l_layer = ConvLayer(l_layer, num_filters=512, filter_size=3, pad=1)
    l_layer = ConvLayer(l_layer, num_filters=512, filter_size=3, pad=1)
    l_layer = PoolLayer(l_layer, pool_size=3, stride=3, ignore_border=False)
    l_layer = DenseLayer(l_layer, num_units=4096)
    l_layer = DropoutLayer(l_layer, p=0.5)
    l_layer = DenseLayer(l_layer, num_units=4096)
    l_layer = DropoutLayer(l_layer, p=0.5)
    return l_layer


  @classmethod
  def _DefaultCNNMaxPoolConfigs(cls):
    return [
      cls.CNNMaxPoolConfig(32, (5,5), (2,2)),
      cls.CNNMaxPoolConfig(32, (5,5), (2,2)),
    ]
