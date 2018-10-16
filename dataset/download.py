import urllib.request
import os
from time import sleep
from glob import glob
from datetime import timedelta
import time
import multiprocessing
from multiprocessing import Process,Queue

###### MULIT THREADER
def multi_thread(procs,max_procs=2):
    prcos_left = len(procs)
    start_time = time.time()
    spin = 0
    # Now for each line try to download the file and save it in the directory
    spins = ['|', '/', '-', '\\'] # spinner to show that things are working
    try:
        for p in procs:
            while(len(multiprocessing.active_children()) > max_procs):
                pass
            prcos_left -= 1
            p.start()
            print('\r{}'.format(spins[spin]),end='') # print out
            spin += 1 # for the spinner
            if(spin > 3):
                spin = 0
        while(len(multiprocessing.active_children()) > 0):
            pass
        print("Procs complete")
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            p.join() # join the boys
    elapsed_time = time.time() - start_time
    seconds = timedelta(seconds=int(round(elapsed_time)))
    print("Time elapsed: " + str(seconds))

# Download a file from a url
def down_url(url, file_name):
    try:
        urllib.request.urlretrieve(url, file_name)
    except Exception as ex:
        pass

# main function
def setup_folder(url_list, folder_name):
    print('Downloading -> {}'.format(folder_name))
    # first make our save directory
    os.makedirs(folder_name, exist_ok=True)
    # a url list will be like this: <class name>_url.txt
    # - images will be labled like so, <class name>_<example number>
    # get our class label
    class_name = url_list.split('_')[0]
    idx = 0
    spin = 0
    # Now for each line try to download the file and save it in the directory
    spins = ['|', '/', '-', '\\'] # spinner to show that things are working
    procs = [] # for storing all the processes that will be done
    with open(url_list, 'r') as file:
        for line in file: # go through the file
            line = line.replace('\n', '')
            ext = line.split('/')[-1].split('.')[-1]
            if(not os.path.exists('{}{}_{}.{}'.format(folder_name,class_name,idx,ext))):
                procs.append(Process(target=down_url, args=(line, '{}{}_{}.{}'.format(folder_name,class_name,idx,ext),))) # download
            idx += 1
    
        file.close()
    # use the multi threader to process all of our procs
    multi_thread(procs, max_procs=4)

if __name__ == '__main__':
    text_files = glob('*.txt') # get all of out url lists
    for text_file in text_files:# for each of them run the downloader
        folder = text_file.split('_')[0]
        setup_folder(text_file, '{}/'.format(folder))
