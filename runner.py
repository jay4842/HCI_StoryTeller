import os
import sys
import argparse

# custom imports
import src.downloader as down
import src.train.train as train

data_dir = 'data/'
restore_path = 'final/'

parser = argparse.ArgumentParser(description='')
parser.add_argument('--epochs', dest='epochs', type=int ,default=50, help='The number of epochs preformed while training.')
parser.add_argument('--batch_size', dest='batch_size', type=int ,default=50, help='The size of your batch for each training step.')
parser.add_argument('--data_path', dest='data_path',default='data/cifar-10/', help='Where our data starts.')
parser.add_argument('--save_dir', dest='save_dir',default='model_saves/', help='Where we save our model.')
parser.add_argument('--learning_rate', dest='learning_rate', type=float ,default=0.0002, help='Set our learning rate for the optimizer.')
parser.add_argument('--test_in', dest='test_in', default=data_dir + 'cifar-10/test/', help='What our test input is.')
parser.add_argument('--ext', dest='ext', default='png', help='What extention images have.')
parser.add_argument('--num_classes', dest='num_classes', default=10, type=int, help='How many classes in your dataset.')
parser.add_argument('--train', dest='train', type=bool ,default=False, help='Set runner to train.')
parser.add_argument('--test', dest='test', type=bool ,default=False, help='Set runner to test.')
parser.add_argument('--run', dest='run', default='ice_01', help='What we save our run as.')
parser.add_argument('--mode', dest='mode', default='classify', help='Pick which network to run.')

# rnn specific stuff here
# more later
args = parser.parse_args()


if __name__ == '__main__':
    if not(os.path.exists('data/cifar-10/')):
        down.get_cifar_10(save_dir='data/cifar-10/')
    
    if(args.train and args.mode=='classify'):
        train.train(args, image_size=[64,64])



