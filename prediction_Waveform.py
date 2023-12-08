import sounddevice as sd
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from scipy.io.wavfile import write

# Constants
fs = 44100  # Sample rate
seconds = 2  # Recording duration
filename = "prediction.wav"  # Temporary audio file
class_names = ["etc", "gaesaekki", "shibal"]  # Your class names

# Loading the saved model
model = load_model("saved_model/best_model_waveform.h5")

print("Prediction Started: ")
while True:
    print("Say Now: ")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()
    write(filename, fs, myrecording)
    
    # Extracting waveform features
    audio, _ = librosa.load(filename, sr=fs)
    
    # Reshape for CNN
    input_feature = audio[np.newaxis, ..., np.newaxis]

    # Predicting the class
    prediction = model.predict(input_feature)
    
    # Converting prediction to class label
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)
    
    print(f"Predicted class: {predicted_class}, Confidence: {confidence}")
