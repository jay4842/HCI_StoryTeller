import urllib
import os

# Download a file from a url
def down_url(url, file_name):
    try:
        urllib.request.urlretrieve(url, file_name)
    except Exception as ex:
        print('Error occured {}'.format(ex))

# main function
def setup_folder(url_list, folder_name):
    # first make our save directory
    os.makedirs(folder_name, exist_ok=True)
    # a url list will be like this: <class name>_url.txt
    # - images will be labled like so, <class name>_<example number>
    # get our class label
    class_name = url_list.split('_')[0]
    idx = 0

    # Now for each line try to download the file and save it in the directory
    with open(url_list, 'r') as file:
        for line in file:
            line = line.replace('\n', '')
            down_url(line, '{}_{}.jpg'.format(class_name,idx))
            idx += 1
            input('->') # place holder


if __name__ == '__main__':
    setup_folder('bunny_url.txt', 'bunny/')
    



