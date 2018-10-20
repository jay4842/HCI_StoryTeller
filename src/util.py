import cv2
from glob import glob
import os

# This guy will open all images in a folder and see what images are usable
# - path should be like: folder/
# - This will just open all the items in folder
# - This also infers that they are all images
def inspect_folder(path):
    # spinner stuff
    spin = 0
    spins = ['|', '/', '-', '\\']
    # first get all the paths
    paths = glob(path + '*')
    print('Checking {} images'.format(len(paths)))
    # now loop through and open each images, check for None
    working_paths = []
    for path in paths:
        img = None
        try:
            img = cv2.imread(path)
        except Exception as ex:
            pass
        if(img is not None):
            path = path.split('/')
            path = '{}/{}'.format(path[7],path[8])
            working_paths.append(path)
        print('\r{}'.format(spins[spin]),end='')
        spin += 1 # for the spinner
        if(spin > 3):
            spin = 0;
    print('\n{} working paths'.format(len(working_paths)))
    return working_paths

if __name__ == '__main__':
    save_list = '/home/jay/BINA/HCI_StoryTeller/list/'
    os.makedirs(save_list, exist_ok=True)

    folders = ['bird', 'bunny', 'cat', 'cow', 'cup', 'dog', 'fish', 'person', 'sloth', 'toy']
    start_path = '/media/jay/Elements/BINA/HCI_StoryTeller/dataset/'
    for folder in folders:
        paths = inspect_folder(start_path + folder + '/')
        # now make the path file
        with open(save_list + folder + '.lst', 'w') as file:
            for line in paths:
                file.write('{}\n'.format(line))
            file.close()



