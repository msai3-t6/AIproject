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

# Loading our saved model
model = load_model("saved_model/best_model_Mel-S.h5")

print("Prediction Started: ")
while True:
    print("Say Now: ")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()
    write(filename, fs, myrecording)

    # Extracting mel-spectrogram features
    audio, sample_rate = librosa.load(filename)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=2048, hop_length=512, n_mels=128)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    # Reshape for CNN
    input_feature = log_mel_spectrogram[np.newaxis, ..., np.newaxis]

    # Predicting the class
    prediction = model.predict(input_feature)
    
    # Converting prediction to class label
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)
    
    print(f"Predicted class: {predicted_class}, Confidence: {confidence}")
