import os

import random

def GetFilePathsUnderDir(dir_path, shuffle=True):
  file_names = os.listdir(dir_path)
  if shuffle:
    random.shuffle(os.listdir(dir_path))
  for file_name in file_names:
    file_path = os.path.join(dir_path, file_name)
    if os.path.isfile(file_path):
      yield file_path
