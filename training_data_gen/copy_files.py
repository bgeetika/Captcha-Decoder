import os
import sys
import shutil

def copy_files(source, dst):
    print "starting copying files"
    list_of_files = os.listdir(source)
    for filename in list_of_files[0:1000000]:
        filename = os.path.join(source, filename)
        shutil.copy(filename, dst)

source = sys.argv[1]
dst = sys.argv[2]
copy_files(source, dst)

