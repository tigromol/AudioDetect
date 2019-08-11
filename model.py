from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import sys
import record as rec


#Parse argvs and load or record the sample
if len(sys.argv)>1:
    positive_sample,_ = librosa.core.load(sys.argv[1])
else:
    print("*"*50 + "Input duration in seconds" + "*"*50)
    duration = int(input())
    positive_sample = rec.recordSampleToFind(duration).reshape((duration * 44100))
negative_data = np.load("./dataset/X.npy")

#data augmentation
def stretch(data, rate=1):
    input_length = len(data)
    data = librosa.effects.time_stretch(data, rate)
    if len(data)>input_length:
        data = data[:input_length]
    else:
        data = np.pad(data, (0, max(0, input_length - len(data))), "constant")

    return data
 
def augment_data(sample, count):
  dataset = []
  mel_sample = librosa.feature.melspectrogram(sample,n_mels=40)
  for _ in range(int(count/3)):
    dataset.append(librosa.power_to_db(librosa.feature.melspectrogram(sample + np.random.randn(len(sample))*0.005*np.random.randint(1,3),n_mels=40)))
    
    dataset.append(librosa.power_to_db(librosa.feature.melspectrogram(stretch(sample,np.random.uniform(0.7,1.3)),n_mels=40)))
    rnd = np.random.randint(0,negative_data.shape[0])
    random_neg = negative_data[rnd]
    rnd = np.random.randint(0,random_neg.shape[1]-mel_sample.shape[1])
    dataset.append(mel_sample + random_neg[:,rnd:rnd+ mel_sample.shape[1]])
  
  return np.array(dataset)
print ("*"*50 + "Choose number of positive samples (maximum 80)" + "*"*50)
num = int(input())
positive_data = augment_data(positive_sample,num)


def dataset(positive, negative, ratio=5):
  if positive.shape[2] > negative.shape[2]:
    negative = np.roll(negative,positive.shape[2] - negative.shape[2],2)
  elif negative.shape[2] > positive.shape[2]:
    rnd = np.random.randint(0,negative.shape[2]-positive.shape[2])
    negative = negative[:,:,rnd:rnd + positive.shape[2]]
  np.random.shuffle(negative)
  negative = negative[:positive.shape[0]*ratio,:,:]
  label = np.concatenate((np.array([np.array([1,0]) for x in range(positive.shape[0]*ratio)]), np.array([np.array([0,1]) for x in range(positive.shape[0])])))
  dataset = np.concatenate((negative,positive))
  return dataset, label


dataset, label = dataset(positive_data, negative_data)
dataset = dataset.reshape((dataset.shape[0],dataset.shape[1],dataset.shape[2],1))


model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=dataset.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
print("choose number os epochs ,  10 recommended")
epo = input()
model.fit(dataset,label,epochs=int(epo),shuffle=True,batch_size=int(dataset.shape[0]/int(epo)))

model.save("./model/model")
print("Model trained and saved")