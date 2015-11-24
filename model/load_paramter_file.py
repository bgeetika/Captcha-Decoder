import numpy as np


def load_file():
  filename = "/home/sujeetb/geetika/dataset/clipping_more_lstm_2015_11_23_15_41_45.npy.npz"
  filename_new = "/home/sujeetb/geetika/dataset/clipping_bidirec_more_lstm_2015_11_23_15_58_27.npy.npz"
  file_path = "/home/sujeetb/geetika/dataset/new_file.npy.npz"
  with np.load(filename) as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    print len(f.files), len(param_values)
    del f.f
  with np.load(filename_new) as f:
    param_values_new = [f['arr_%d' % i] for i in range(len(f.files))]
    print len(f.files), len(param_values_new)
    del f.f
  x = 0
  while(x < len(param_values_new)):
    '''
    if param_values_new[x].shape != param_values[x].shape:
      if len(param_values_new[x].shape) == 2:
         print param_values_new[x].shape, param_values[x].shape
         rows, cols = param_values[x].shape
         rows_new, cols_new = param_values_new[x].shape
         rows_updates = min(rows,rows_new)
         cols_updates = min(
         #assert(param_values_new[x].shape[1] > param_values[x].shape[1])     
         #assert(param_values_new[x].shape[0] >= param_values[x].shape[0])     
         param_values_new[x][0:256,0:cols] = param_values[x][0:256,0:cols]
      else:
        print "here"
        rows = param_values[x].shape[0]
        assert(param_values_new[x].shape[0] >= param_values[x].shape[0])
        param_values_new[x][0:rows] = param_values[x]
     '''
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



