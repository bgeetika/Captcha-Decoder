from PIL import Image

from StringIO import StringIO
import model
from model.captcha_cracker import CaptchaCracker
import os
import theano
theano.config.floatX = "float64"
import os
from PIL import Image
import numpy
import random
from training_data_gen.image_preprocessor import ImagePreprocessor

def read_and_parse(file_content,cracker):
    im = read_data(file_content)
    array = numpy.asarray(im.convert('L')).copy() 
    image_input = ImagePreprocessor.ProcessImage(array)
    #print "saving image"
    #im.save("geet.png")
    predicted_chars, char_probabilities = cracker.InferForImageArray(image_input)
    return "".join(x for x in predicted_chars)

def read_data(file_content):
    im = Image.open(StringIO(file_content))
    return im
