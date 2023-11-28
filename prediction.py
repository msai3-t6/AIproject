######## IMPORTS ##########
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np
from tensorflow.keras.models import load_model

####### ALL CONSTANTS #####
fs = 44100
seconds = 2
filename = "prediction.wav"
class_names = ["down", "go", "left", "no", "right", "stop", "up", "yes"]  # Update class_names

##### LOADING OUR SAVED MODEL and PREDICTING ###
model = load_model("saved_model/WWD_vproject.h5")

print("Prediction Started: ")
i = 0
while True:
    print("Say Now: ")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()
    write(filename, fs, myrecording)

    audio, sample_rate = librosa.load(filename)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=60) #40에서 60
    mfcc_processed = np.mean(mfcc.T, axis=0)

    prediction = model.predict(np.expand_dims(mfcc_processed, axis=0))
    predicted_index = np.argmax(prediction)  # Get the index of the highest probability

    print(f"Predicted class: {class_names[predicted_index]}")