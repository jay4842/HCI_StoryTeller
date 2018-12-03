import os
import sys
import argparse
import nltk

# custom imports
import src.downloader as down
import src.classify.train as train
import src.classify.test as test
import src.text.rnn_train as rnn_train
import src.text.rnn_test as rnn_test
import src.text.txt_download as txt_down
import src.deploy as deploy

# TODO: add download other text data
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
parser.add_argument('--test_dir', dest='test_dir', default='test/', help='Where you want to save test results.')
parser.add_argument('--restore', dest='restore', default=None, help='Point to where our model save is.')
parser.add_argument('--download_cifar100', dest='download_cifar100', type=bool, default=False, help='Download cifar-100 data set')
parser.add_argument('--download_cifar10', dest='download_cifar10', type=bool, default=False, help='Download cifar-10 data set')

# text download stuff
parser.add_argument('--download_poe', dest='download_poe', type=bool, default=False, help='If we want to download poe data.')
parser.add_argument('--download_nltk', dest='download_nltk', type=bool, default=False, help='Set to true if you need to download the nltk data.')
# rnn specific stuff here
parser.add_argument('--author', dest='author', default='poe', help='What author to be used for testing.')
parser.add_argument('--test_size', dest='test_size', type=int, default=2000, help='How many characters to generate during testing.')

# deploy arguments here
parser.add_argument('--deploy', dest='deploy', type=bool, default=False, help='Run the deploy phase. [Make sure your models saves are setup before using this]')

# more later
args = parser.parse_args()


if __name__ == '__main__':
    if (not(os.path.exists('data/cifar-10/')) or args.download_cifar10):
        down.get_cifar_10(save_dir='data/cifar-10/')
    os.system('clear') # clear screen

    #cifar-100 stuff
    if (args.download_cifar100):
        down.get_cifar_100(save_dir='data/cifar-100/')
    os.system('clear') # clear screen

    if(args.download_nltk):
        nltk.download('all')

    # training
    if(args.train):
        if(args.mode=='classify'):
            train.train(args, image_size=[64,64])
        elif(args.mode=='rnn'):
            rnn_train.train_rnn(args)
    # testing
    if(args.test):
        if(args.mode=='classify'):
            test.run_test(args)
        elif(args.mode=='rnn'):
            rnn_test.test_rnn(args)
    # download poe
    if(args.download_poe):
        txt_down.download_poe()
    # run deploy
    if(args.deploy):
        deploy.deploy(args)

