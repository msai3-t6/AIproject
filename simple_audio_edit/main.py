import numpy as np
import librosa
import datetime

from tensorflow.keras import models
import os 
from recording_helper import record_audio, terminate
from tf_helper import preprocess_audiobuffer
from recording_helper import preprocess_audiobuffer
from scipy.io.wavfile import write

# !! Modify this in the correct order
labels = ['gaesaekki', 'shibal']
loaded_model = models.load_model("saved_model")
threshold_db =  -30
fs = 16000


os.makedirs('recording', exist_ok=True)

print("Prediction Started: ")
while True:
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    print(f"Say Now {current_time}: ")
    audio = record_audio()

    ###### 수정필요 오디오 소리가 작을 경우 녹음되지 않도록 
    amplitudes = np.abs(audio)
    max_amplitude = np.max(amplitudes)
    if max_amplitude < 30: 
        print("음성의 소리가 작아 녹음되지 않았습니다.")
        continue

    spec = preprocess_audiobuffer(audio)
    prediction = loaded_model(spec)


    max_probability = np.max(prediction)
    if max_probability < 0.90:
        print("예측 확률이 너무 낮아 예측하지 않았습니다.")
        continue

    filename = f"recording/prediction_{current_time}.wav"
    write(filename, fs, audio)

    predicted_index = np.argmax(prediction, axis=1)
    command = labels[predicted_index[0]]
    print("Predicted label:", command)

    # if prediction[0, predicted_index] > 0.90:
    #     print(f"'{labels[predicted_index]}' 단어가 검출되었습니다({i})")

def predict_mic():
    audio = record_audio()
    spec = preprocess_audiobuffer(audio)
    prediction = loaded_model(spec)
    predicted_index = np.argmax(prediction, axis=1)
    command = labels[predicted_index[0]]
    print("Predicted label:", command)
    return command



if __name__ == "__main__":
    while True:
        command = predict_mic()
