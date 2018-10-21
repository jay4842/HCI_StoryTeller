import cv2
import numpy as np
from glob import glob
import os
import tensorflow as tf

# Making a tf session with some gpu options too
def get_session(gpu_fraction=.5):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''
    num_threads = 2
    # gives error that has to deal with the version of tensorflow, and the cudNN version as well
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    #return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    config = tf.ConfigProto() #allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = False
    config.gpu_options.per_process_gpu_memory_fraction=gpu_fraction=gpu_fraction
    config.intra_op_parallelism_threads=num_threads=num_threads
    #config.log_device_placement=True
    sess = tf.Session(config=config)
    return sess


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
        except Exception:
            pass
        if(img is not None):
            path = path.split('/')
            path = '{}/{}'.format(path[7],path[8])
            working_paths.append(path)
        print('\r{}'.format(spins[spin]),end='')
        spin += 1 # for the spinner
        if(spin > 3):
            spin = 0
    print('\n{} working paths'.format(len(working_paths)))
    return working_paths

# Get test images loaded
def get_test_image(image_path, label_split=None, classes=None, 
                    w=128, h=256, standardize=True, channel_swap=[2,1,0]):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (w, h))
    if(standardize):
        image = (image - np.mean(image)) / (np.std(image))
    #
    if(label_split is not None and classes is not None):
        label = image_path.split('/')[-1].split(label_split)[0]
        label = classes.index(label)
        bin_label = [0 for x in range(len(classes))]
        bin_label[label] = 1
        return image, bin_label
    
    return image

# return a batch of images, and if label_info is not not none return labels too
def get_images(batch_paths,ids,label_split=None, classes=None, 
               w=128, h=256, standardize=True, channel_swap=[2,1,0]):
    images = []
    labels = []
    for idx, b in enumerate(ids):
        path_ = batch_paths[b]
        path_ = path_.replace('\\', '/')
        image = cv2.imread(path_)
        image = cv2.resize(image, (w,h))
        # random crop it
        if(label_split is not None and classes is not None):
            label = path_.split('/')[-1].split(label_split)[0]
            label = classes.index(label)
            bin_label = [0 for x in range(len(classes))]
            bin_label[label] = 1
            labels.append(bin_label)
        if np.random.random() > 0.5:
            image = np.fliplr(image)

        # normalize the boys
        #image = normalize(image)
        if(standardize):
            image = (image - np.mean(image)) / (np.std(image))
        images.append(image)

    # check if there are labels too
    if(len(labels) > 0):
        return images, labels
    else:
        return images


def get_batch(paths, ids, batch_size=5, label_split=None, classes=None, standardize=True, w=128, h=256):
    batch_ids = np.random.choice(ids,batch_size)  
    return get_images(paths, batch_ids, label_split=label_split, classes=classes, standardize=standardize, w=w, h=h)

# the other uses the same process but adds a few extra steps
def sigmoid_cross_entropy_balanced(logits, label, name='cross_entropy_loss'):
    """
       Implements Equation [2] in https://arxiv.org/pdf/1504.06375.pdf
       Compute edge pixels for each training sample and set as pos_weights to
       tf.nn.weighted_cross_entropy_with_logits
    """

    # Fix the issue with the incompatible shapes
    #  Curtusy of dugan
    #tf.image.resize_nearest_neighbor(label,(1000,500))

    #diff_w = (b.shape[2].value - 500)/2
    #diff_h = (b.shape[1].value - 1000)/2
    #b = b[:, diff_h:b.shape[1].value-diff_h, diff_w:b.shape[2].value-diff_w,:]

    #print('logits {0}  label {1}'.format(logits,label))
    y = tf.cast(label, tf.float32)
        
    count_neg = tf.reduce_sum(1 - y)
    count_pos = tf.reduce_sum(y)

    # Equation [2]
    beta = count_neg / (count_neg + count_pos)

    # Equation [2] divide by 1 - beta
    pos_weight = beta / (1-beta)

    #print(label.shape)
    #print(logits.shape)
    #input('')
    cost = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=y, pos_weight=pos_weight)

    # multiply by 1 - beta
    cost = tf.reduce_mean(cost * (1-beta))

    # check if image has no edge pixels, return 0 else return complete error function
    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost, name=name)
    # end of model helpers


# MAIN IF WE WANT TO RUN FROM HERE
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



