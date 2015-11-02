import argparse
import sys


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("TrainingDirc", help="Path to training data directory")
    parser.add_argument("ValidateDirc", help="Path to Validation data file")
    parser.add_argument("TestDirc", help="Path to test data file")
    parser.add_argument("ModelParamsFile", help="Path to the directory where parans is to be stored followed by prefix for the parameter file")
    parser.add_argument("-maxsoft","--maxsoft", help="provide this argument if you want to run maxsoft")
    parser.add_argument("-bidirec", "--bidirec", help="Provide this argument in order to run bidirectional lstm", action="store_true")
    parser.add_argument("-hiddenlayers","--hiddenlayers", help="number of hidden layers in the network")
    parser.add_argument("-learningrate", "--learningrate", help="learning rate")
    parser.add_argument("-batchsize", "--batchsize", help="learning rate")
    parser.add_argument("-testsize", "--testsize", help="learning rate")
    parser.add_argument("-includeCapital", "--includeCapital", help="include capital letters or not",action="store_true" )
    parser.add_argument("-length", "--length", help="length of the characters")
    args = parser.parse_args()
    print "Params passed are : "
    print args
    print type(args)
    return args
