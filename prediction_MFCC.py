import os
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np
from tensorflow.keras.models import load_model

# Constants
fs = 44100
seconds = 2
filename = "prediction.wav"
class_names = ["etc", "gaesaekki", "shibal"]
num_columns = 173  # this should be the max_length from your preprocessing step

# Loading our saved model
model = load_model("saved_model/best_model_MFCC.h5")

print("Prediction Started: ")
while True:
    print("Say Now: ")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, myrecording)

    # Extracting MFCC features
    audio, sample_rate = librosa.load(filename, mono=True, sr=44100)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)

    # Adjust the length of the MFCC to match the expected input shape
    if mfcc.shape[1] > num_columns:
        mfcc = mfcc[:, :num_columns]
    else:
        pad_width = num_columns - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')


    # Reshape for Conv1D
    input_feature = mfcc[np.newaxis, ...]

    # Predicting the class
    prediction = model.predict(input_feature)
    
    # Converting prediction to class label
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)
    
    print(f"Predicted class: {predicted_class}, Confidence: {confidence}")
