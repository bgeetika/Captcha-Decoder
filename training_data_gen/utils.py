import os

def GetFilePathsUnderDir(dir_path):
  for file_name in os.listdir(dir_path):
    file_path = os.path.join(dir_path, file_name)
    if os.path.isfile(file_path):
      yield file_path
