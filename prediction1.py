######## IMPORTS ##########
import librosa
import numpy as np
from tensorflow.keras.models import load_model

####### ALL CONSTANTS #####
filename = "yes/100.wav"  # Update with your file path
class_names = ["down", "go", "left", "no", "right", "stop", "up", "yes"]  # Update class_names

##### LOADING OUR SAVED MODEL and PREDICTING ###
model = load_model("saved_model/WWD_vproject.h5")

print("Prediction Started: ")

audio, sample_rate = librosa.load(filename)
mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
mfcc_processed = np.mean(mfcc.T, axis=0)

prediction = model.predict(np.expand_dims(mfcc_processed, axis=0))
predicted_index = np.argmax(prediction)  # Get the index of the highest probability

print(f"Predicted class: {class_names[predicted_index]}")