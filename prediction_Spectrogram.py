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
num_rows = 1025  # this should be the row size of your spectrogram
num_columns = 173  # this should be the max_length from your preprocessing step

# Loading our saved model
model = load_model("saved_model/best_model_spectrogram.h5")

# Function to extract spectrogram
def extract_spectrogram(file_path, num_rows, num_columns):
    y, sr = librosa.load(file_path, sr=44100)
    D = librosa.stft(y)  # STFT of y
    spect = librosa.amplitude_to_db(np.abs(D), ref=np.max)  # Convert to dB scale

    # Adjust the size of the spectrogram to match the expected input shape
    if spect.shape[1] > num_columns:
        spect = spect[:, :num_columns]
    else:
        pad_width = num_columns - spect.shape[1]
        spect = np.pad(spect, pad_width=((0, 0), (0, pad_width)), mode='constant')

    return spect

print("Prediction Started: ")
while True:
    print("Say Now: ")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, myrecording)

    # Extracting spectrogram features
    spect = extract_spectrogram(filename, num_rows, num_columns)

    # Reshape for CNN
    input_feature = spect[np.newaxis, ..., np.newaxis]

    # Predicting the class
    prediction = model.predict(input_feature)
    
    # Converting prediction to class label
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)
    
    print(f"Predicted class: {predicted_class}, Confidence: {confidence}")
