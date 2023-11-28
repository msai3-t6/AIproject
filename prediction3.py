######## IMPORTS ##########
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np
from tensorflow.keras.models import load_model

####### ALL CONSTANTS #####
fs = 44100
seconds = 3
filename = "prediction.wav"
class_names = ["down", "go", "left", "no", "right", "stop", "up", "yes"]  # Update class_names

##### LOADING OUR SAVED MODEL and PREDICTING ###
model = load_model("saved_model/WWD_vproject.h5")

print("Prediction Started: ")
i = 0
# threshold 변수 정의
threshold = 0.1  # 적절한 값으로 변경하세요

while True:
    print("Say Now: ")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()
    
    # 추가: 음성이 특정 임계값 이상인 경우에만 파일로 저장
    if np.max(np.abs(myrecording)) > threshold:
        # 시간 느림/빠름 변형
        time_stretched_recording = librosa.effects.time_stretch(myrecording[:, 0], rate=1.5)
        write(filename, fs, np.column_stack((time_stretched_recording, time_stretched_recording)))
    else:
        print("Too quiet. Try again.")
        continue

    audio, sample_rate = librosa.load(filename)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=60)
    mfcc_processed = np.mean(mfcc.T, axis=0)

    prediction = model.predict(np.expand_dims(mfcc_processed, axis=0))
    predicted_index = np.argmax(prediction)  # Get the index of the highest probability

    print(f"Predicted class: {class_names[predicted_index]}")