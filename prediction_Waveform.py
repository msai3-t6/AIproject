import sounddevice as sd
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from scipy.io.wavfile import write

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
num_rows = 88200  # this should be the max_length from your preprocessing step

# Loading our saved model
model = load_model("saved_model/best_model_waveform.h5")

# Function to extract waveform
def extract_waveform(file_path, num_rows):
    waveform, sr = librosa.load(file_path, sr=44100)

    # Adjust the size of the waveform to match the expected input shape
    if len(waveform) > num_rows:
        waveform = waveform[:num_rows]
    else:
        pad_width = num_rows - len(waveform)
        waveform = np.pad(waveform, pad_width=(0, pad_width), mode='constant')

    return waveform

print("Prediction Started: ")
while True:
    print("Say Now: ")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, myrecording)

    # Extracting waveform features
    waveform = extract_waveform(filename, num_rows)

    # Reshape for Conv1D
    input_feature = waveform[np.newaxis, ..., np.newaxis]

    # Predicting the class
    prediction = model.predict(input_feature)
    
    # Converting prediction to class label
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)
    
    print(f"Predicted class: {predicted_class}, Confidence: {confidence}")


# # Constants
# fs = 44100  # Sample rate
# seconds = 2  # Recording duration
# filename = "prediction.wav"  # Temporary audio file
# class_names = ["etc", "gaesaekki", "shibal"]  # Your class names

# # Loading the saved model
# model = load_model("saved_model/best_model_waveform.h5")

# print("Prediction Started: ")
# while True:
#     print("Say Now: ")
#     myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
#     sd.wait()
#     write(filename, fs, myrecording)
    
#     # Extracting waveform features
#     audio, _ = librosa.load(filename, sr=fs)
    
#     # Reshape for CNN
#     input_feature = audio[np.newaxis, ..., np.newaxis]

#     # Predicting the class
#     prediction = model.predict(input_feature)
    
#     # Converting prediction to class label
#     predicted_class = class_names[np.argmax(prediction)]
#     confidence = np.max(prediction)
    
#     print(f"Predicted class: {predicted_class}, Confidence: {confidence}")
