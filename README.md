# speech-recognition

Spoken Digit Recognition
In this notebook, You will do Spoken Digit Recognition.

Input - speech signal, output - digit number

It contains

1)Reading the dataset. and Preprocess the data set. Detailed instrctions are given below. You have to write the code in the same cell which contains the instrction.

2)Training the LSTM with RAW data

3)Converting to spectrogram and Training the LSTM network

4)Creating the augmented data and doing step 2 and 3 again.

Preprocessing

All files are in the "WAV" format. We will read those raw data files using the librosa

Based on our analysis 99 percentile values are less than 0.8sec so we will limit maximum length of X_train_processed and X_test_processed to 0.8 sec. It is similar to pad_sequence for a text dataset.

While loading the audio files, we are using sampling rate of 22050 so one sec will give array of length 22050. so, our maximum length is 0.8*22050 = 17640 Pad with Zero if length of sequence is less than 17640 else Truncate the number.

Also create a masking vector for train and test.

masking vector value = 1 if it is real value, 0 if it is pad value. Masking vector data type must be bool.

## 1. Giving Raw data directly.
Now we have

Train data: X_train_pad_seq, X_train_mask and y_train
Test data: X_test_pad_seq, X_test_mask and y_test

We will create a LSTM model which takes this input.

Task:

Create an LSTM network which takes "X_train_pad_seq" as input, "X_train_mask" as mask input. You can use any number of LSTM cells. Please read LSTM documentation(https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM) in tensorflow to know more about mask and also https://www.tensorflow.org/guide/keras/masking_and_padding

Get the final output of the LSTM and give it to Dense layer of any size and then give it to Dense layer of size 10(because we have 10 outputs) and then compile with the sparse categorical cross entropy( because we are not converting it to one hot vectors). Also check the datatype of class labels(y_values) and make sure that you convert your class labels to integer datatype before fitting in the model.

While defining your model make sure that you pass both the input layer and mask input layer as input to lstm layer as follows 

Use tensorboard to plot the graphs of loss and metric(use custom micro F1 score as metric) and histograms of gradients. You can write your code for computing F1 score using this link

make sure that it won't overfit.

You are free to include any regularization

## 2. Converting into spectrogram and giving spectrogram data as input
We can use librosa to convert raw data into spectrogram. A spectrogram shows the features in a two-dimensional representation with the intensity of a frequency at a point in time i.e we are converting Time domain to frequency domain. you can read more about this in https://pnsn.org/spectrograms/what-is-a-spectrogram

Now we have

Train data: X_train_spectrogram and y_train
Test data: X_test_spectrogram and y_test

We will create a LSTM model which takes this input.

Task:

Create an LSTM network which takes "X_train_spectrogram" as input and has to return output at every time step.

Average the output of every time step and give this to the Dense layer of any size. (ex: Output from LSTM will be (None, time_steps, features) average the output of every time step i.e, you should get (None,time_steps) and then pass to dense layer )

give the above output to Dense layer of size 10( output layer) and train the network with sparse categorical cross entropy.

Use tensorboard to plot the graphs of loss and metric(use custom micro F1 score as metric) and histograms of gradients. You can write your code for computing F1 score using this link

make sure that it won't overfit.

You are free to include any regularization

## 3. Data augmentation with raw features
Till now we have done with 2000 samples only. It is very less data. We are giving the process of generating augmented data below.

There are two types of augmentation:

time stretching - Time stretching either increases or decreases the length of the file. For time stretching we move the file 30% faster or slower
pitch shifting - pitch shifting moves the frequencies higher or lower. For pitch shifting we shift up or down one half-step.

Follow the steps
Split data 'df_audio' into train and test (80-20 split)

We have 2000 data points(1600 train points, 400 test points)

Do augmentation only on X_train,pass each point of X_train to generate_augmented_data function.After augmentation we will get 14400 train points. Make sure that you are augmenting the corresponding class labels (y_train) also.
Preprocess your X_test using load_wav function.

Convert the augmented_train_data and test_data to numpy arrays.

Perform padding and masking on augmented_train_data and test_data.

After padding define the model similar to model 1 and fit the data

## 4. Data augmentation with spectogram data
use convert_to_spectrogram and convert the padded data from train and test data to spectogram data.

The shape of train data will be 14400 x 64 x 35 and shape of test_data will be 400 x 64 x35

Define the model similar to model 2 and fit the data

Note - While fitting your model on the augmented data for model 3 you might face Resource exhaust error. One simple hack to avoid that is save the augmented_train_data,augment_y_train,test_data and y_test to Drive or into your local system. Then restart the runtime so that now you can train your model with full RAM capacity. Upload these files again in the new runtime session perform padding and masking and then fit your model.

