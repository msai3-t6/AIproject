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
class_names = ["etc", "gaesaekki", "shibal"]  # Update class_names

##### LOADING OUR SAVED MODEL and PREDICTING ###
model = load_model("saved_model/myproject_modified_cnn.h5") # update model name

print("Prediction Started: ")
i = 0

# threshold 변수 정의
threshold = 0.1
prob_threshold = 0.5  # 적절한 값으로 변경하세요

while True:
    print("Say Now: ")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()

    if np.max(np.abs(myrecording)) > threshold:
        write(filename, fs, myrecording)
    else:
        print("Too quiet. Try again.")
        continue

    audio, sample_rate = librosa.load(filename)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfcc_processed = np.mean(mfcc.T, axis=0)

    prediction = model.predict(np.expand_dims(mfcc_processed, axis=0))
    max_prob = np.max(prediction)  # Get the highest probability

    # 확률이 일정 수준 이상일 때만 예측 클래스 출력
    if max_prob > prob_threshold:
        predicted_index = np.argmax(prediction)
        print(f"Predicted class: {class_names[predicted_index]}, Probability: {max_prob}")
    else:
        print("Not sure. Try again.")