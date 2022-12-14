# -*- coding: utf-8 -*-
"""Speech detection Assignment2022.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1r42HkkVh2QprFz9lIVMEBmMW78HJWMIM

# <font color='red'> Spoken Digit Recognition</font>

In this notebook, You will do Spoken Digit Recognition. 

Input - speech signal, output - digit number



It contains  

1. Reading the dataset. and Preprocess the data set. Detailed instrctions are given below. You have to write the code in the same cell which contains the instrction. 
2. Training the LSTM with RAW data
3. Converting to spectrogram and Training the LSTM network
4. Creating the augmented data and doing step 2 and 3 again.  

<font size=5>Instructions:</font>

    1. Don't change any Grader Functions. Don't manipulate any Grader functions. If you manipulate any, it will be considered as plagiarised. 
    
    2. Please read the instructions on the code cells and markdown cells. We will explain what to write. 
    
    3. Please return outputs in the same format what we asked. Eg. Don't return List of we are asking for a numpy array.
    
    4. Please read the external links that we are given so that you will learn the concept behind the code that you are writing.
    
    5. We are giving instructions at each section if necessary, please follow them. 

<font size=5>Every Grader function has to return True. </font>
"""

import numpy as np
import pandas as pd
import librosa
import os
from sklearn.metrics import f1_score
##if you need any imports you can do that here.

"""We shared recordings.zip, please unzip those. """

#read the all file names in the recordings folder given by us
#(if you get entire path, it is very useful in future)
#save those files names as list in "all_files"
#https://stackoverflow.com/questions/49685924/extract-google-drive-zip-from-google-colab-notebook
!unzip /content/recordings.zip

#all_files = os.listdir('recordings')
#!/usr/bin/python

import os, sys

# Open a file
path = "recordings"
all_files = os.listdir( path )

"""<font size=4>Grader function 1 </font>"""

def grader_files():
    temp = len(all_files)==2000
    temp1 = all([x[-3:]=="wav" for x in all_files])
    temp = temp and temp1
    return temp
grader_files()

"""Create a dataframe(name=df_audio) with two columns(path, label).   
You can get the label from the first letter of name.  
Eg: 0_jackson_0 --> 0  
0_jackson_43 --> 0

## Exploring the sound dataset
"""

#It is a good programming practise to explore the dataset that you are dealing with. This dataset is unique in itself because it has sounds as input
#https://colab.research.google.com/github/Tyler-Hilbert/AudioProcessingInPythonWorkshop/blob/master/AudioProcessingInPython.ipynb
#visualize the data and write code to play 2-3 sound samples in the notebook for better understanding.
#please go through the following reference video https://www.youtube.com/watch?v=37zCgCdV468





"""## Creating dataframe"""

# extract the labels from all_files
#Create a dataframe(name=df_audio) with two columns(path, label).   
#You can get the label from the first letter of name.  
#Eg: 0_jackson_0 --> 0  
#0_jackson_43 --> 0
z=[ele.split('_') for ele in all_files]
extracted_label=[]
for i,j,k in z:
    #print(i,j,k)
    extracted_label.append(i)

extracted_label[:4]

#Create a dataframe(name=df_audio) with two columns(path, label).   
#You can get the label from the first letter of name.  
#Eg: 0_jackson_0 --> 0  
#0_jackson_43 --> 0
path = "recordings"
all_files = os.listdir(path)
#https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
df_audio = pd.DataFrame()
df_audio['path'] = [path+'/' + i for i in all_files]
df_audio['label'] = extracted_label[:]

df_audio.head()

#df_audio.to_csv('C:\\Users\\mural\\Desktop\\aaic assignments\\speech recognition\\New folder\\file1_speech.csv')

#info
df_audio.info()

"""<font size=4>Grader function 2 </font>"""

def grader_df():
    flag_shape = df_audio.shape==(2000,2)
    flag_columns = all(df_audio.columns==['path', 'label'])
    list_values = list(df_audio.label.value_counts())
    flag_label = len(list_values)==10
    flag_label2 = all([i==200 for i in list_values])
    final_flag = flag_shape and flag_columns and flag_label and flag_label2
    return final_flag
grader_df()

from sklearn.utils import shuffle
df_audio = shuffle(df_audio, random_state=45)#don't change the random state

"""<pre><font size=4>Train and Validation split</font></pre>"""

#split the data into train and validation and save in X_train, X_test, y_train, y_test
#use stratify sampling
#use random state of 45
#use test size of 30%
from sklearn.model_selection import train_test_split
x,y=df_audio['path'],df_audio['label']
X_train, X_test, y_train, y_test=train_test_split(x,y,test_size=0.3,random_state=45,
                                                  stratify=y)

"""<font size=4>Grader function 3 </font>"""

def grader_split():
    flag_len = (len(X_train)==1400) and (len(X_test)==600) and (len(y_train)==1400) and (len(y_test)==600)
    values_ytrain = list(y_train.value_counts())
    flag_ytrain = (len(values_ytrain)==10) and (all([i==140 for i in values_ytrain]))
    values_ytest = list(y_test.value_counts())
    flag_ytest = (len(values_ytest)==10) and (all([i==60 for i in values_ytest]))
    final_flag = flag_len and flag_ytrain and flag_ytest
    return final_flag
grader_split()

"""<pre><font size=4>Preprocessing</font>

All files are in the "WAV" format. We will read those raw data files using the librosa</pre>
"""

sample_rate = 22050
def load_wav(x, get_duration=True):
    '''This return the array values of audio with sampling rate of 22050 and Duration'''
    #loading the wav file with sampling rate of 22050
    samples, sample_rate = librosa.load(x, sr=22050)
    if get_duration:
        duration = librosa.get_duration(samples, sample_rate)
        return [samples, duration]
    else:
        return samples

#samples, sample_rate = librosa.load(str(train_audio_path)+filename)

#use load_wav function that was written above to get every wave. 
#save it in X_train_processed and X_test_processed
# X_train_processed/X_test_processed should be dataframes with two columns(raw_data, duration) with same index of X_train/y_train

#X_train_processed=load_wav(X_train,get_duration=True)

samples_train,duration_train=[],[]

for value in X_train:
    z=load_wav(value, get_duration=True)
    samples_train.append(z[0]) #z[0] means samples
    duration_train.append(z[1]) #z[1] means duration

samples_test,duration_test=[],[]

for value in X_test:
    z=load_wav(value, get_duration=True)
    samples_test.append(z[0]) #z[0] means samples
    duration_test.append(z[1]) #z[1] means duration

#https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
d = {'raw_data' : samples_train,'duration':duration_train}
d1 = {'raw_data' : samples_test,'duration':duration_test}

X_train_processed=pd.DataFrame(data=d)
X_test_processed=pd.DataFrame(data=d1)

#https://www.w3schools.com/python/matplotlib_histograms.asp
#plot the histogram of the duration for trian
import matplotlib.pyplot as plt
plt.hist(X_train_processed['duration'])

# displaying the title
plt.title("Train duration audio")

plt.show()

#plot the histogram of the duration for trian
#https://www.w3schools.com/python/matplotlib_histograms.asp
#plot the histogram of the duration for trian
import matplotlib.pyplot as plt
plt.hist(X_test_processed['duration'])

# displaying the title
plt.title("Test duration audio")

plt.show()

#print 0 to 100 percentile values with step size of 10 for train data duration. 
#percentile cal
#step1 difference=max_val-min_val
#step2 boundary for ex: low=0.25*diff,high=0.75*diff 
#step3  low<values<high 
#diff=max(X_train_processed['duration'])-min(X_train_processed['duration'])
#low=0*diff
#high=1*diff

#perc_list=[]

#for val in X_train_processed['duration']:
#    if val>=low and val<=high:
#        perc_list.append(val)

# Python Program illustrating
# numpy.percentile() method
#https://www.geeksforgeeks.org/numpy-percentile-in-python/
import numpy as np

# 1D array
arr = X_train_processed['duration']
for i in range(11):
    print(str(10*i)+"th percentile of train data duration:",np.percentile(arr, 10*i))

##print 90 to 100 percentile values with step size of 1. 
# Python Program illustrating
# numpy.percentile() method
#https://www.geeksforgeeks.org/numpy-percentile-in-python/
import numpy as np

# 1D array
arr = X_test_processed['duration']
for i in range(90,101):
    print(str(i)+"th percentile of test data duration:",np.percentile(arr,i))

"""<font size=4>Grader function 4 </font>"""

def grader_processed():
    flag_columns = (all(X_train_processed.columns==['raw_data', 'duration'])) and (all(X_test_processed.columns==['raw_data', 'duration']))
    flag_shape = (X_train_processed.shape ==(1400, 2)) and (X_test_processed.shape==(600,2))
    return flag_columns and flag_shape
grader_processed()

"""<b>Based on our analysis 99 percentile values are less than 0.8sec so we will limit maximum length of X_train_processed and X_test_processed to 0.8 sec. It is similar to pad_sequence for a text dataset.</b>

<b>While loading the audio files, we are using sampling rate of 22050 so one sec will give array of length 22050. so, our maximum length is 0.8*22050 = 17640
</b>
<b>Pad with Zero if length of sequence is less than 17640 else Truncate the number. </b>

<b> Also create a masking vector for train and test. </b>

<b> masking vector value = 1 if it is real value, 0 if it is pad value. Masking vector data type must be bool.</b>

"""

max_length  = 17640

## as discussed above, Pad with Zero if length of sequence is less than 17640 else Truncate the number. 
## save in the X_train_pad_seq, X_test_pad_seq
## also Create masking vector X_train_mask, X_test_mask

## all the X_train_pad_seq, X_test_pad_seq, X_train_mask, X_test_mask will be numpy arrays mask vector dtype must be bool.

#https://stackoverflow.com/questions/42598630/why-cant-i-use-preprocessing-module-in-keras
import tensorflow as tf
import keras
from keras import preprocessing
from tensorflow.keras.preprocessing.sequence import pad_sequences

train_sequences=X_train_processed.raw_data
test_sequences=X_test_processed.raw_data
max_length  = 17640

X_train_pad_seq=   preprocessing.sequence.pad_sequences(train_sequences,maxlen=max_length ,
                                                dtype = 'float32',padding='post',truncating='post')

X_test_pad_seq=   preprocessing.sequence.pad_sequences(test_sequences,maxlen=max_length ,
                                                dtype = 'float32',padding='post',truncating='post')

#masking vector
#https://www.youtube.com/watch?v=sjIzRpVXd30
train_mask = X_train_pad_seq>0
test_mask  = X_test_pad_seq>0

X_train_mask=train_mask
X_test_mask=test_mask

X_train_mask

#X_train_pad_seq1 = np.expand_dims(X_train_pad_seq, -1)
#X_test_pad_seq1 = np.expand_dims(X_test_pad_seq, -1)

"""<font size=4>Grader function 5 </font>"""

def grader_padoutput():
    flag_padshape = (X_train_pad_seq.shape==(1400, 17640)) and (X_test_pad_seq.shape==(600, 17640)) and (y_train.shape==(1400,))
    flag_maskshape = (X_train_mask.shape==(1400, 17640)) and (X_test_mask.shape==(600, 17640)) and (y_test.shape==(600,))
    flag_dtype = (X_train_mask.dtype==bool) and (X_test_mask.dtype==bool)
    return flag_padshape and flag_maskshape and flag_dtype
grader_padoutput()

"""### 1. Giving Raw data directly.

Now we have

Train data: X_train_pad_seq, X_train_mask and y_train  
Test data: X_test_pad_seq, X_test_mask and y_test   

We will create a LSTM model which takes this input. 

Task:

1. Create an LSTM network which takes "X_train_pad_seq" as input, "X_train_mask" as mask input. You can use any number of LSTM cells. Please read LSTM documentation(https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM) in tensorflow to know more about mask and also https://www.tensorflow.org/guide/keras/masking_and_padding 
2. Get the final output of the LSTM and give it to Dense layer of any size and then give it to Dense layer of size 10(because we have 10 outputs) and then compile with the sparse categorical cross entropy( because we are not converting it to one hot vectors). Also check the datatype of class labels(y_values) and make sure that you convert your class labels  to integer datatype before fitting in the model.
3. While defining your model make sure that you pass both the input layer and mask input layer as input to lstm layer as follows
<img src='https://i.imgur.com/FvcgvbY.jpg'>
4. Use tensorboard to plot the graphs of loss and metric(use custom micro F1 score as metric) and histograms of gradients. You can write your code for computing F1 score using this <a  href='https://i.imgur.com/8YULUcu.jpg'>link</a> 

5. make sure that it won't overfit. 
6. You are free to include any regularization
"""

y_train = y_train.values.astype('int')
y_test = y_test.values.astype('int')

X_train_pad_seq =X_train_pad_seq
X_test_pad_seq = X_test_pad_seq

"""Call backs"""

#https://towardsdatascience.com/custom-callback-functions-for-transformers-ae65e30c094f
#https://imgur.com/8YULUcu
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import f1_score

class Metrics(Callback):
    def __init__(self, train_data,validation_data):
        self.validation_data = validation_data
        self.train_data = train_data
    def on_epoch_end(self, epoch, logs={}):
        
        train_predict = np.argmax(tf.nn.softmax(self.model.predict(self.train_data[0])),1)
        train_targ = self.train_data[1]
        train_f1 = f1_score(train_targ, train_predict, average='micro')
        logs['train_f1'] = train_f1


        val_predict = np.argmax(tf.nn.softmax(self.model.predict(self.validation_data[0])),1)
        val_targ = self.validation_data[1]
        val_f1 = f1_score(val_targ, val_predict, average='micro')
        logs['val_f1'] = val_f1

        print (f'??? val_f1: {round(val_f1,4)} ')
        print (f'??? train_f1: {round(train_f1,4)} ')
        return val_f1,train_f1

class F1Score(tf.keras.callbacks.Callback):
    def __init__(self, train_data, validation_data):
        super().__init__()
        self.train_data = train_data
        self.validation_data = validation_data
        self.history = {}
        self.history['val_f1_score'] = []
  
    def on_epoch_end(self, epochs, logs = {}):
        train_preds = np.argmax(self.model.predict(self.train_data[0]), axis = 1)
        train_f1_score = f1_score(self.train_data[1], train_preds, average='micro')
        #train_f1_score = np.round(train_f1_score, 4)

        test_preds = np.argmax(self.model.predict(self.validation_data[0]), axis = 1)
        test_f1_score = f1_score(self.validation_data[1], test_preds, average='micro')
        #test_f1_score = np.round(test_f1_score, 4)
        self.history['val_f1_score'].append(test_f1_score)

        print(f" train- f1_score: {train_f1_score} - val_f1_score: {test_f1_score}")

from tensorflow.keras.callbacks import EarlyStopping
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=1)

from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model
import tensorflow as tf

## as discussed above, please write the architecture of the model.
## you will have two input layers in your model (data input layer and mask input layer)
## make sure that you have defined the data type of masking layer as bool

import datetime

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
# Clear any logs from previous runs
!rm -rf ./logs/

#callbacks
log_dir = os.path.join("logs",'fits', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,histogram_freq=1,write_graph=True)



#creating the first model
tf.keras.backend.clear_session()
in_put = Input(shape = (17640,1))  #array of max len is 17640 and having 1 dimension from X_train padded
x_mask = Input(shape = (17640), dtype = 'bool') #mask have 17640 maxlen
layer_1= tf.keras.layers.LSTM(10)(in_put, mask = x_mask)

# Usage in a Keras layer:
#https://keras.io/api/layers/initializers/
GlorotNormal_initializer = tf.keras.initializers.GlorotNormal()
HeNormal_initializer = tf.keras.initializers.HeNormal()


layer_2 = tf.keras.layers.Dense(20, activation = 'relu', kernel_initializer = GlorotNormal_initializer)(layer_1)
out = tf.keras.layers.Dense(10, activation = 'softmax', kernel_initializer = HeNormal_initializer)(layer_2)

model1 = Model(inputs = [in_put, x_mask], outputs = out)
#printing the summary of model
model1.summary()

#train your model
#model1.fit([X_train_pad_seq,X_train_mask],y_train_int,.........)

#https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
#Adam Configuration Parameters
#alpha. Also referred to as the learning rate or step size. 
#The proportion that weights are updated (e.g. 0.001). 
#Larger values (e.g. 0.3) results in faster initial learning before the rate is updated. 
#Smaller values (e.g. 1.0E-5) slow learning right down during training

model1.compile(optimizer = tf.keras.optimizers.Adam(0.001), loss = 'sparse_categorical_crossentropy')

callback = [F1Score(([X_train_pad_seq, X_train_mask], y_train),([X_test_pad_seq, X_test_mask], y_test)),
      earlystop,tensorboard_callback]

model1.fit([X_train_pad_seq, X_train_mask], y_train, validation_data = ([X_test_pad_seq, X_test_mask], y_test),
           batch_size = 80, epochs = 2, callbacks = callback)

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir logs/fits

"""### 2. Converting into spectrogram and giving spectrogram data as input  

We can use librosa to convert raw data into spectrogram. A spectrogram shows the features in a two-dimensional representation with the
intensity of a frequency at a point in time i.e we are converting Time domain to frequency domain. you can read more about this in https://pnsn.org/spectrograms/what-is-a-spectrogram

"""

def convert_to_spectrogram(raw_data):
    '''converting to spectrogram'''
    spectrum = librosa.feature.melspectrogram(y=raw_data, sr=sample_rate, n_mels=64)
    logmel_spectrum = librosa.power_to_db(S=spectrum, ref=np.max)
    return logmel_spectrum

##use convert_to_spectrogram and convert every raw sequence in X_train_pad_seq and X_test_pad-seq.
## save those all in the X_train_spectrogram and X_test_spectrogram ( These two arrays must be numpy arrays)
#reducing the dimensions back
#https://www.geeksforgeeks.org/using-numpy-to-convert-array-elements-to-float-type/
#https://stackoverflow.com/questions/49586458/parametererror-data-must-be-floating-point-librosa
#convert dtype to float otherwise it shows audio should be in floating point

X_train_spectrogram = []
for raw_ele in X_train_pad_seq:
    X_train_spectrogram.append(convert_to_spectrogram(raw_ele))
X_train_spectrogram = np.array(X_train_spectrogram)

X_test_spectrogram = []
for raw_ele in X_test_pad_seq:
    X_test_spectrogram.append(convert_to_spectrogram(raw_ele))
X_test_spectrogram = np.array(X_test_spectrogram)

"""<font size=4>Grader function 6 </font>"""

def grader_spectrogram():
    flag_shape = (X_train_spectrogram.shape==(1400,64, 35)) and (X_test_spectrogram.shape == (600, 64, 35))
    return flag_shape
grader_spectrogram()

"""
Now we have

Train data: X_train_spectrogram and y_train  
Test data: X_test_spectrogram and y_test   

We will create a LSTM model which takes this input. 

Task:

1. Create an LSTM network which takes "X_train_spectrogram" as input and has to return output at every time step. 
2. Average the output of every time step and give this to the Dense layer of any size. 
(ex: Output from LSTM will be  (None, time_steps, features) average the output of every time step i.e, you should get (None,time_steps) 
and then pass to dense layer )
3. give the above output to Dense layer of size 10( output layer) and train the network with sparse categorical cross entropy.  
4. Use tensorboard to plot the graphs of loss and metric(use custom micro F1 score as metric) and histograms of gradients. You can write your code for computing F1 score using this <a  href='https://i.imgur.com/8YULUcu.jpg'>link</a> 
5. make sure that it won't overfit. 
6. You are free to include any regularization
"""

#https://classroom.appliedroots.com/v2/faqs/3DALB9LQ/
# write the architecture of the model
#print model.summary and make sure that it is following point 2 mentioned above
tf.keras.backend.clear_session()
in_put = Input(shape = (64,35,))
layer_1 = LSTM(192, return_sequences = True)(in_put)
#https://www.tensorflow.org/api_docs/python/tf/math/reduce_mean
#https://stackoverflow.com/questions/65185733/how-to-apply-average-pooling-at-each-time-step-of-lstm-output
layer_2 = tf.reduce_mean(layer_1, axis = -1)

# Usage in a Keras layer:
#https://keras.io/api/layers/initializers/
GlorotNormal_initializer = tf.keras.initializers.GlorotNormal()
HeNormal_initializer = tf.keras.initializers.HeNormal()
layer_3 = Dense(192, activation = 'relu', kernel_initializer = GlorotNormal_initializer)(layer_2)
out_put = Dense(10, activation = 'softmax', kernel_initializer = GlorotNormal_initializer )(layer_3)

model2 = Model(inputs = in_put, outputs = out_put)
#printing the model summary
model2.summary()

#compile and fit your model.
#model2.fit([X_train_spectrogram],y_train_int,......)

class F1ScoreCB(tf.keras.callbacks.Callback):
    def __init__(self, train_data, test_data):
        super().__init__()
        self.train_data = train_data
        self.test_data = test_data
        self.history = {}
        self.history['val_f1_score'] = []
  
    def on_epoch_end(self, epochs, logs = {}):
        train_preds = np.argmax(self.model.predict(self.train_data[0]), axis = -1)
        train_f1_score = f1_score(self.train_data[1], train_preds, average='micro')
        train_f1_score = np.round(train_f1_score, 4)

        test_preds = np.argmax(self.model.predict(self.test_data[0]), axis = -1)
        test_f1_score = f1_score(self.test_data[1], test_preds, average='micro')
        test_f1_score = np.round(test_f1_score, 4)
        self.history['val_f1_score'].append(test_f1_score)

        print(f" - f1_score: {train_f1_score} - val_f1_score: {test_f1_score}")

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
# Clear any logs from previous runs
!rm -rf ./logs/

Metrics((X_train_spectrogram, y_train),(X_test_spectrogram, y_test))

#https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
#Adam Configuration Parameters
#alpha. Also referred to as the learning rate or step size. 
#The proportion that weights are updated (e.g. 0.001). 
#Larger values (e.g. 0.3) results in faster initial learning before the rate is updated. 
#Smaller values (e.g. 1.0E-5) slow learning right down during training

#https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam
model2.compile(optimizer = tf.keras.optimizers.Adam(0.0005), loss = 'sparse_categorical_crossentropy')
callback = [F1Score((X_train_spectrogram, y_train), (X_test_spectrogram, y_test)),
      earlystop,tensorboard_callback]

model2.fit(X_train_spectrogram, y_train, validation_data = (X_test_spectrogram, y_test),
           batch_size = 16, epochs = 20, callbacks = callback)

#model2.compile(optimizer = tf.keras.optimizers.Adam(0.0006), loss = 'sparse_categorical_crossentropy')
#cb = [Metrics((X_train_spectrogram, y_train),(X_test_spectrogram, y_test)),
      #earlystop,tensorboard_callback]

#model2.fit(X_train_spectrogram, y_train, validation_data = (X_test_spectrogram, y_test),
           #batch_size = 16, epochs = 15, callbacks = cb)

# Commented out IPython magic to ensure Python compatibility.

# %tensorboard --logdir logs/fits

"""### 3. Data augmentation with raw features 

Till now we have done with 2000 samples only. It is very less data. We are giving the process of generating augmented data below.

There are two types of augmentation:
1. time stretching - Time stretching either increases or decreases the length of the file. For time stretching we move the file 30% faster or slower
2. pitch shifting - pitch shifting moves the frequencies higher or lower. For pitch shifting we shift up or down one half-step.

"""

## generating augmented data. 
def generate_augmented_data(file_path):
    augmented_data = []
    samples = load_wav(file_path,get_duration=False)
    for time_value in [0.7, 1, 1.3]:
        for pitch_value in [-1, 0, 1]:
            time_stretch_data = librosa.effects.time_stretch(samples, rate=time_value)
            final_data = librosa.effects.pitch_shift(time_stretch_data, sr=sample_rate, n_steps=pitch_value)
            augmented_data.append(final_data)
    return augmented_data

"""## Follow the steps 

1. Split data 'df_audio' into train and test (80-20 split)

2. We have 2000 data points(1600 train points, 400 test points) 


"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(df_audio['path'],df_audio['label'],random_state = 32321,test_size=0.2,stratify=df_audio['label'])

"""3. Do augmentation only on X_train,pass each point of X_train to generate_augmented_data function.After augmentation we will get 14400 train points. Make sure that you are augmenting the corresponding class labels (y_train) also.
4. Preprocess your X_test using load_wav function.
5. Convert the augmented_train_data and test_data to numpy arrays.
6. Perform padding and masking on augmented_train_data and test_data.
7. After padding define the model similar to model 1 and fit the data

<font color='red'> Note </font> - While fitting your model on the augmented data for model 3 you might face Resource exhaust error. One simple hack to avoid that is save the augmented_train_data,augment_y_train,test_data and y_test to Drive or into your local system. Then restart the runtime so that now you can train your model with full RAM capacity. Upload these files again in the new runtime session perform padding and masking and then fit your model.
"""

#Data augmentation on train data and preprocessing on test data
train_augemented_data = []
train_augemented_labels = []

for path, label in zip(X_train.values, y_train.values):
    aug_data = generate_augmented_data(path)
    #train_augemented_data.extend(aug_data)#https://www.programiz.com/python-programming/methods/list/extend
    #append() adds a single element to the end of the list while . extend() can add multiple individual elements to the end of the list
    train_augemented_data+=aug_data
    train_augemented_labels.extend(label*9)#https://www.programiz.com/python-programming/methods/list/extend

X_train_processed = pd.DataFrame({'raw_data' : train_augemented_data, 'label' : train_augemented_labels})

#X_train_processed=pd.read_csv('/content/final_preprocessed_sound_model3_train1.csv')

#X_test_processed=pd.read_csv('/content/final_preprocessed_sound_model3_test.csv')

test_data,test_duration=[],[]

for path in X_test:
    z=load_wav(path, get_duration=True)
    test_data.append(z[0]) #z[0] means samples
    test_duration.append(z[1]) #z[1] means duration

X_test_processed = pd.DataFrame({'raw_data' : test_data, 'label' : y_test})

#https://stackoverflow.com/questions/42598630/why-cant-i-use-preprocessing-module-in-keras
import tensorflow as tf
import keras
from keras import preprocessing
from tensorflow.keras.preprocessing.sequence import pad_sequences

train_sequences=X_train_processed.raw_data
test_sequences=X_test_processed.raw_data
max_length  = 17640

X_train_pad_seq3=   preprocessing.sequence.pad_sequences(X_train_processed.raw_data,maxlen=max_length ,
                                                dtype = 'float32',padding='post',truncating='post')

X_test_pad_seq3=   preprocessing.sequence.pad_sequences(X_test_processed.raw_data,maxlen=max_length ,
                                                dtype = 'float32',padding='post',truncating='post')

#saving the dataframe
#X_train_processed.to_csv(r'final_preprocessed_sound_model3_train1.csv')
#X_test_processed.to_csv(r'final_preprocessed_sound_model3_test.csv')
#from google.colab import files
#files.download('final_preprocessed_sound_model3_train1.csv')
#files.download('final_preprocessed_sound_model3_test.csv')

y_train3 = X_train_processed.label.values.astype('int')
y_test3 = X_test_processed.label.values.astype('int')

X_train_mask3 = X_train_pad_seq3>0
X_test_mask3 = X_test_pad_seq3>0

#creating the first model
tf.keras.backend.clear_session()
in_put = Input(shape = (17640,1))  #array of max len is 17640 and having 1 dimension from X_train padded
x_mask = Input(shape = (17640), dtype = 'bool') #mask have 17640 maxlen
layer_1= tf.keras.layers.LSTM(10)(in_put, mask = x_mask)

# Usage in a Keras layer:
#https://keras.io/api/layers/initializers/
GlorotNormal_initializer = tf.keras.initializers.GlorotNormal()
HeNormal_initializer = tf.keras.initializers.HeNormal()


layer_2 = tf.keras.layers.Dense(20, activation = 'relu', kernel_initializer = GlorotNormal_initializer)(layer_1)
out = tf.keras.layers.Dense(10, activation = 'softmax', kernel_initializer = GlorotNormal_initializer)(layer_2)

model3 = Model(inputs = [in_put, x_mask], outputs = out)
#printing the summary of model
model3.summary()

#tf.keras.backend.clear_session()
#inp = Input(shape = (17640,1))
#inp_mask = Input(shape = (17640), dtype = 'bool')

#x = LSTM(10)(inp, mask = inp_mask)

#x = Dense(20, activation = 'relu', kernel_initializer = 'he_normal')(x)
#out = Dense(10, activation = 'softmax', kernel_initializer = 'glorot_normal')(x)

#model3 = Model(inputs = [inp, inp_mask], outputs = out)



model3.compile(optimizer = tf.keras.optimizers.Adam(0.001), loss = 'sparse_categorical_crossentropy')

callback = [F1Score(([X_train_pad_seq3, X_train_mask3], y_train3), ([X_test_pad_seq3, X_test_mask3], y_test3)),
      earlystop,tensorboard_callback]

model3.fit([X_train_pad_seq3, X_train_mask3], y_train3, validation_data = ([X_test_pad_seq3, X_test_mask3], y_test3),
           batch_size =380, epochs = 10, callbacks = callback)

# Commented out IPython magic to ensure Python compatibility.

# %tensorboard --logdir logs/fits

"""### 4. Data augmentation with spectogram data

1. use convert_to_spectrogram and convert the padded data from train and test data to spectogram data.
2. The shape of train data will be 14400 x 64 x 35 and shape of test_data will be 400 x 64 x35
3. Define the model similar to model 2 and fit the data
"""

def convert_to_spectrogram(raw_data):
    '''converting to spectrogram'''
    spectrum = librosa.feature.melspectrogram(y=raw_data, sr=sample_rate, n_mels=64)
    logmel_spectrum = librosa.power_to_db(S=spectrum, ref=np.max)
    return logmel_spectrum


X_train_spectrogram4 = []
for raw_ele in X_train_pad_seq3:
    X_train_spectrogram4.append(convert_to_spectrogram(raw_ele))
X_train_spectrogram4 = np.array(X_train_spectrogram4)

X_test_spectrogram4 = []
for raw_ele in X_test_pad_seq3:
    X_test_spectrogram4.append(convert_to_spectrogram(raw_ele))
X_test_spectrogram4 = np.array(X_test_spectrogram4)

y_train4 = y_train3
y_test4 = y_test3

#https://classroom.appliedroots.com/v2/faqs/3DALB9LQ/
# write the architecture of the model
#print model.summary and make sure that it is following point 2 mentioned above
tf.keras.backend.clear_session()
in_put = Input(shape = (64,35,))
layer_1 = LSTM(192, return_sequences = True)(in_put)
#https://www.tensorflow.org/api_docs/python/tf/math/reduce_mean
#https://stackoverflow.com/questions/65185733/how-to-apply-average-pooling-at-each-time-step-of-lstm-output
layer_2 = tf.reduce_mean(layer_1, axis = -1)

# Usage in a Keras layer:
#https://keras.io/api/layers/initializers/
GlorotNormal_initializer = tf.keras.initializers.GlorotNormal()
HeNormal_initializer = tf.keras.initializers.HeNormal()
layer_3 = Dense(192, activation = 'relu', kernel_initializer = GlorotNormal_initializer)(layer_2)
out_put = Dense(10, activation = 'softmax', kernel_initializer = GlorotNormal_initializer )(layer_3)

model4 = Model(inputs = in_put, outputs = out_put)
#printing the model summary
model4.summary()

# Commented out IPython magic to ensure Python compatibility.

from tensorflow.keras.callbacks import EarlyStopping
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=1)


import datetime

# %load_ext tensorboard
# Clear any logs from previous runs
!rm -rf ./logs/ 

#callbacks
log_dir = os.path.join("logs",'fits', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,histogram_freq=1,write_graph=True)

#https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam
model4.compile(optimizer = tf.keras.optimizers.Adam(0.0005), loss = 'sparse_categorical_crossentropy')
cb = [F1ScoreCB((X_train_spectrogram4, y_train4), (X_test_spectrogram4, y_test4)),
      earlystop,tensorboard_callback]

model4.fit(X_train_spectrogram4, y_train4, validation_data = (X_test_spectrogram4, y_test4),
           batch_size = 16, epochs = 10, callbacks = cb)

# Commented out IPython magic to ensure Python compatibility.

# %tensorboard --logdir logs/fits

"""#Inferences:

1) In model 1 and model 3 as the data does not convert into time domain frequency so that model does not learn and generate a low f1 score.

2)In model 2 and model 4 as the data converted into Time Domain frequency spectograms so that model learns and genearte a high f1 score.
Also average of time step frequency reduce the dimensionality and it reaches optimality in less Time.

3)Data augmentation helps to model do not overfitting on the data.

"""



