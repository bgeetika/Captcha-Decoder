from PIL import Image
from resizeimage import resizeimage
import glob

def resize_file(in_file, out_file, size):
    with Image.open(in_file) as fd:
        new_width, new_height = size
        fd = fd.resize((new_width, new_height), Image.ANTIALIAS)
    fd.save(out_file)
    fd.close()


for filename in glob.glob('/home/geetika/captcha/dataset_ssd_1T/new_dataset_website/*.jpg'):
    resize_file(filename, filename, (200, 50))


