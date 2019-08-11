from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import load_model
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
    audio,_ = librosa.core.load(sys.argv[1])
else:
    print("input duration in seconds")
    duration = int(input())
    audio = rec.recordAudioToDetect(duration).reshape((duration * 44100))

model = load_model("./model/model")
raw_audio = audio
audio = librosa.feature.melspectrogram(audio,n_mels=40)
print(audio.shape)
def sliding_window(audio,  model):
  errors = []
  for i in range(audio.shape[1]-model.input_shape[2]):
    tmp = audio[:,i:i+model.input_shape[2]].reshape(1,model.input_shape[1],model.input_shape[2],1)
    errors.append(model.predict(tmp))
  return errors

errors = sliding_window(audio, model)
plt.subplot(212)
plt.plot([np.argmax(x) for x in errors])
plt.subplot(211)
plt.plot(raw_audio)
plt.show()