# Story Teller

# requirments
- scipy==1.0.0
- SpeechRecognition==3.8.1
- termcolor==1.1.0
- tensorflow==1.4.1
- requests==2.14.2
- opencv_python==3.3.0.10
- numpy==1.14.0
- nltk==3.4
- Pillow==5.3.0
- beautifulsoup4==4.6.3
- gtts==2.0.1
- Linux based OS (ex: ubuntu 16.04 || raspbian)
- portaudio (brew portaudio)

## requirments continued:
First for this package you will have to install it ([PyAudio](http://people.csail.mit.edu/hubert/pyaudio/#downloads)).  
Next you can install the requirements using pip.  
  `pip install -r requirments.txt`  

# Drive
- [folder](https://drive.google.com/drive/folders/1fwhkkxTkv1GhdQuMBgT72--PmvSuGzkZ?usp=sharing)
Here there is the video as well as some other project resoruces that are not in this repo.  

# RNN datasets
- [Edgar Allan Poe](http://www.textfiles.com/etext/AUTHORS/POE/)
- [Kaggle Text Dataset](https://www.kaggle.com/mylesoneill/classic-literature-in-ascii)

# Network notes
See this link for more information the VGG16 implementation we are using.  
- [vgg16 model info](https://www.cs.toronto.edu/~frossard/post/vgg16/)


# Running
After setting up the requirments you will need to download data.  
If you wish to train the RNN you will need to provide it with data, you can download the [kaggle](https://www.kaggle.com/mylesoneill/classic-literature-in-ascii) dataset for this.  
Place this dataset into the text_dataset folder and extract it there.  
To download the model weights for vgg16 use:  
  `python runner.py --download_vgg=True`  
  
Next you will also need the NLTK data as well:  
  `python runner.py --download_nltk=True`  
  
## Training
To train the RNN you will just need to provide where it is saving data to as well as the mode and train indicator.  
  `python runner.py --train=True --mode=rnn --author=doyle`  

## Testing
To run the deploy the program use this:  
  `python runner.py --deploy=True`  
  
Here you will see a GUI to run different tests. You can also be able to capture images and select  
images to run the testing flow with. Additionally you can change the author you are using as too.  
  
# Future work
- Add more authors  
- Better post processing  
