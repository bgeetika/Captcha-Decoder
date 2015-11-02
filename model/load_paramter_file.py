import numpy as np


def load_file():
  filename = "/home/geetika/captcha/dataset_ssd_1T/complex_mix_dataset_nvocab/lstm_complex_mix_2015_11_01_01_05_23.npy.npz"
  filename_new = "/home/geetika/captcha/dataset_ssd_1T/complex_mix_dataset_nvocab/lstm_complexMix_2015_11_01_14_42_47.npy.npz"
  file_path = "/home/geetika/captcha/dataset_ssd_1T/complex_mix_dataset_nvocab/new_file.npy.npz"
  with np.load(filename) as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    print len(f.files), len(param_values)
    del f.f
  with np.load(filename_new) as f:
    param_values_new = [f['arr_%d' % i] for i in range(len(f.files))]
    print len(f.files), len(param_values)
    del f.f
  x = 0
  while(x < len(param_values_new)):
    if param_values_new[x].shape != param_values[x].shape:
      if len(param_values_new[x].shape) == 2:
         rows, cols = param_values[x].shape
         assert(param_values_new[x].shape[1] > param_values[x].shape[1])     
         assert(param_values_new[x].shape[0] >= param_values[x].shape[0])     
         param_values_new[x][0:rows,0:cols] = param_values[x]
      else:
        rows = param_values[x].shape[0]
        assert(param_values_new[x].shape[0] >= param_values[x].shape[0])
        param_values_new[x][0:rows] = param_values[x]
    else:
       param_values_new[x] = param_values[x]
    x += 1
  
  np.savez(file_path, *param_values_new)

  with np.load(file_path) as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    print len(f.files), len(param_values)
    for x in param_values:
        print x.shape
    del f.f
  
       
load_file()



