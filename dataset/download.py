import urllib.request
import os
from time import sleep
from glob import glob

# Download a file from a url
def down_url(url, file_name):
    try:
        urllib.request.urlretrieve(url, file_name)
    except Exception as ex:
        pass

# main function
def setup_folder(url_list, folder_name):
    # first make our save directory
    os.makedirs(folder_name, exist_ok=True)
    # a url list will be like this: <class name>_url.txt
    # - images will be labled like so, <class name>_<example number>
    # get our class label
    class_name = url_list.split('_')[0]
    idx = 0
    spin = 0
    # Now for each line try to download the file and save it in the directory
    spins = ['|', '/', '-', '\\']

    with open(url_list, 'r') as file:
        for line in file:
            line = line.replace('\n', '')
            down_url(line, '{}{}_{}.jpg'.format(folder_name,class_name,idx))
            sleep(0.05) # sleep for .05 of a second. Dont 
            idx += 1
            print('\r{}'.format(spins[spin]),end='')
            spin += 1
            if(spin > 3):
                spin = 0;

if __name__ == '__main__':
    text_files = glob('*.txt')
    for text_file in text_files:
        folder = text_file.split('_')[0]
        setup_folder(text_file, '{}/'.format(folder))




