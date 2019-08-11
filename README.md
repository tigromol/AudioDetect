# AudioDetect
Detect audio sample via CNN

To train model on script you need to execute model.py script:
if you want to use sample you need to pass it's path as argument like this:
python model.py "./path/to/wav.wav"
else script gonna record sample from audio device
after training, model saved in ./model dir

To predict where is audio sample in audio you need to execute predict.py scripy:
if you want to use audio you need to pass it's path as argument like this:
python model.py "./path/to/wav.wav"
else script gonna record audio from audio device
after running , script plot two plots , one with original audio , second with predicted classes (1 for sample, 0 for none).

X.npy is Material licensed by TUT .
The Database is accessible at the following address: https://www.kaggle.com/c/acoustic-scene-2018 .
