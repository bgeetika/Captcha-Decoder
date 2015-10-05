import lasagne
import numpy
import theano
import theano.tensor as T

import training_data_gen.vocabulary as vocabulary

class Model(object):
  def __init__(self, saved_params_path=None):
    self._network, self._train_fn, self._test_fn =  self._Initialize()
    if saved_params_path:
      with numpy.load(saved_params_path) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(self._network, param_values)

  def SaveModelParamsToFile(self, file_path):
    numpy.savez(file_path, *lasagne.layers.get_all_param_values(self._network))

  def GetTrainFn(self):
    return self._train_fn

  def GetTestFn(self):
    return self._test_fn

  class CNNMaxPoolConfig(object):
    def __init__(self, num_cnn_filters, cnn_filter_size, max_pool_size):
      self.num_cnn_filters = num_cnn_filters
      self.cnn_filter_size = cnn_filter_size
      self.max_pool_size = max_pool_size

  @classmethod
  def _Initialize(cls):
    image_input = T.ftensor4('image_input')
    prediction_layer = cls._BuildModel(image_input)

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
	loss, params, learning_rate=0.01, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(prediction_layer, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
							    target_char)
    test_loss = test_loss.mean()
    # An expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_char),
		      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([image_input, target_chars], loss, updates=updates,
                               allow_input_downcast=True)

    # Compile a second function computing the prediction, validation loss and accuracy:
    test_fn = theano.function([image_input, target_chars],
			      [test_prediction, test_loss, test_acc],
                              allow_input_downcast=True)
    return prediction_layer, train_fn, test_fn


  @classmethod
  def _BuildModel(cls,
                  image_input,
                  cnn_max_pool_configs=None,
                  cnn_dense_layer_sizes=[256]):
    if cnn_max_pool_configs is None:
      cnn_max_pool_configs = cls._DefaultCNNMaxPoolConfigs()
    input_shape = T.shape(image_input)
    network = lasagne.layers.InputLayer(shape=(None, 3, 50, 200),
                                        input_var=image_input)
    network = cls._BuildCNN(network, cnn_max_pool_configs, cnn_dense_layer_sizes)

    # And, finally, the softmax layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=len(vocabulary.CHARS),
            nonlinearity=lasagne.nonlinearities.softmax)
    return network


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
  def _DefaultCNNMaxPoolConfigs(cls):
    return [
      cls.CNNMaxPoolConfig(16, (5,5), (2,2)),
      cls.CNNMaxPoolConfig(8, (5,5), (2,2)),
      cls.CNNMaxPoolConfig(4, (5,5), (2,2)),
    ]
