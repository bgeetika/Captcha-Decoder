import argparse
import sys


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("TrainingDirc", help="Path to training data directory")
    parser.add_argument("ValidateDirc", help="Path to Validation data file")
    parser.add_argument("TestDirc", help="Path to test data file")
    parser.add_argument("ModelParamsFile", help="Path to the directory where params is to be stored followed by prefix for the parameter file")
    parser.add_argument("-maxsoft","--maxsoft", help="Provide this argument if you want to run maxsoft", metavar='\b')
    parser.add_argument("-bidirec", "--bidirec", help="Provide this argument in order to run bidirectional lstm", action="store_true")
    parser.add_argument("-hiddenlayers","--hiddenlayers", help="Number of hidden layers in the network", metavar='\b')
    parser.add_argument("-learningrate", "--learningrate", help="Learning rate", metavar='\b')
    parser.add_argument("-batchsize", "--batchsize", help="Training batch size", metavar='\b')
    parser.add_argument("-testsize", "--testsize", help="Test batch size", metavar='\b')
    parser.add_argument("-includeCapital", "--includeCapital", help="Include capital letters or not",action="store_true" )
    parser.add_argument("-length", "--length", help="Length of the characters", metavar='\b')
    parser.add_argument("-rescale_in_preprocessing", "--rescale", help="Rescale_in_preprocessing",action="store_true")
    parser.add_argument("-use_mask_input", "--use_mask_input", help="Use_mask_input", action="store_true")
    parser.add_argument("-lstm_layer_units", "--lstm_layer_units", metavar='\b', help="No of units in lstm layer", type=int, default=256)
    parser.add_argument("-cnn_dense_layer_sizes", "--cnn_dense_layer_sizes", metavar='\b', help="No of units in dense layer", type=int, default=256)
    parser.add_argument("-multichar", "--multichar", help="multichar disabled",action="store_false")
    parser.add_argument("-lstm_grad_clipping", "--lstm_grad_clipping", help="lstm_grad_clipping")
    args = parser.parse_args()
    print "Params passed are : "
    print args
    print type(args)
    return args
