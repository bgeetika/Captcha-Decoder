from BaseHTTPServer import BaseHTTPRequestHandler
import urlparse
import inference 
import json
import model
import base64
from model.captcha_cracker import CaptchaCracker
import os
import theano
theano.config.floatX = "float64"
import os
from PIL import Image
import numpy
import random

lstm_model_params_prefix = ("/home/sujeetb/geetika/Captcha_cracker/rest_server/lstm_variable_run_2015_11_15_22_01_04.npy.npz")
cracker = model.captcha_cracker.CaptchaCracker(
lstm_model_params_prefix, includeCapital=False, multi_chars=True,
rescale_in_preprocessing=False, num_rnn_steps=6, use_mask_input=True)

class GetHandler(BaseHTTPRequestHandler):
      def do_POST(self):
        self.send_response(200)
        self.end_headers()
        content_len = int(self.headers.getheader('content-length')) 
        post_body = self.rfile.read(content_len)
        print type(post_body)
        data = json.loads(post_body)
        utf8_file_content = data['file_content'].encode('utf-8')
        print 'utf8 string', len(utf8_file_content), utf8_file_content[:10]
        #file_content = data['file_content'].encode('ascii','ignore')
        file_content = base64.b64decode(data['file_content'].encode('utf-8'))
        print type(file_content),len(file_content),file_content[:10]
        result = inference.read_and_parse(file_content,cracker)    
        #result =  "testing"
        print result
        self.wfile.write(result)      
        return

if __name__ == '__main__':
    from BaseHTTPServer import HTTPServer
    server = HTTPServer(('0.0.0.0', 8080), GetHandler)
    print 'Starting server, use <Ctrl-C> to stop'
    server.serve_forever()
