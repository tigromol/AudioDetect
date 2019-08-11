import sounddevice as sd
import sys
import numpy as np
import os.path
from scipy.io.wavfile import write

#Settings of recording interface
fs = 44100
sd.default.samplerate = fs
#Mono channel
sd.default.channels = 1
#Default dir for audio
defaultDir = os.path.join(".","audio")


def recordSampleToFind(duration, saveWav=True):
    '''
    Record and save sample to detect
    duration - duration of sample in seconds
    saveWav - if true - save .wav file 
    '''
    filename = "sample"
    myrecording = sd.rec(int(duration * fs))
    sd.wait()
    path = os.path.join(defaultDir, "sample",filename)
    if saveWav:
        write(path + ".wav", fs, myrecording)
    return myrecording
    
def recordAudioToDetect(duration):
    '''
    Record and save audio for detection
    duration - duration of sample in seconds
    '''
    filename = "audio"
    myrecording = sd.rec(int(duration * fs))
    sd.wait()
    path = os.path.join(defaultDir, "audio",filename)
    write(path + ".wav", fs, myrecording)
    return myrecording    

