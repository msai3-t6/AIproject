import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np
from tensorflow.keras.models import load_model

####### ALL CONSTANTS #####
fs = 44100
seconds = 2
filename = "prediction.wav"
class_names = ["etc", "gaesaekki", "shibal"]
threshold = 0.1  # 너무 작은 입력값을 무시하는 임계값
prob_threshold = 0.5  # 예측 클래스를 출력하는 확률 임계값

##### LOADING OUR SAVED MODEL and PREDICTING ###
model = load_model("saved_model/best_model_MFCC.h5")

while True:
    print("Say Now: ")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()

    # 입력값의 최대 절대값이 임계값보다 큰 경우에만 파일로 저장
    if np.max(np.abs(myrecording)) > threshold:
        write(filename, fs, myrecording)
    else:
        print("Too quiet. Try again.")
        continue

    # 오디오 파일에서 MFCC 특징 추출
    audio, sample_rate = librosa.load(filename)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    mfcc_processed = np.mean(mfcc.T, axis=0)[:13]  # 처음 13개의 계수만 사용

    # 모델에 입력하여 예측
    prediction = model.predict(np.expand_dims(mfcc_processed, axis=0))
    max_prob = np.max(prediction)  # 가장 높은 확률값 확인

    # 확률이 일정 수준 이상일 때만 예측 클래스 출력
    if max_prob > prob_threshold:
        predicted_index = np.argmax(prediction)
        print(f"Predicted class: {class_names[predicted_index]}, Probability: {max_prob}")
    else:
        print("Not sure. Try again.")
