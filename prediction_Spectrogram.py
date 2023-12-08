# prediction_spectrogram.py
import os
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np
from keras.models import load_model

# 상수 정의
fs = 44100
seconds = 2
filename = "prediction.wav"
class_names = ["etc", "gaesaekki", "shibal"]

# 저장된 모델 로드
model = load_model("saved_model/best_model_spectrogram.h5")

print("Prediction Started: ")
while True:
    print("Say Now: ")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()
    write(filename, fs, myrecording)

    # 스펙트로그램 특징 추출
    audio, sample_rate = librosa.load(filename)
    spectrogram = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

    # CNN에 입력으로 전달하기 위한 형태 조정
    input_feature = log_spectrogram[np.newaxis, ..., np.newaxis]

    # 클래스 예측
    prediction = model.predict(input_feature)

    # 예측 결과를 클래스 레이블로 변환
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    print(f"Predicted class: {predicted_class}, Confidence: {confidence}")