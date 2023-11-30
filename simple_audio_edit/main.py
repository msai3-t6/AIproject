import numpy as np
import os 
import datetime
from tensorflow.keras import models
from scipy.io.wavfile import write

from recording_helper import record_audio, terminate
from tf_helper_v2 import preprocess_audiobuffer

# !! Modify this in the correct order
labels = ['gaesaekki', 'shibal']
loaded_model = models.load_model("saved_model")
threshold_db =  -50
fs = 16000


os.makedirs('recording', exist_ok=True)


# (11/30) 볼륨 임계값 추가, confidence 추가 해야함 
while True:
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    print(f"Say Now {current_time}: ")
    audio = record_audio()

    spec = preprocess_audiobuffer(audio)
    prediction = loaded_model(spec)

    filename = f"recording/prediction_{current_time}.wav"
    write(filename, fs, audio)

    label_pred = np.argmax(prediction, axis=1)
    command = labels[label_pred[0]]
    print("Predicted label:", command)




# def predict_mic():
#     audio = record_audio()
#     spec = preprocess_audiobuffer(audio)
#     prediction = loaded_model(spec)
#     label_pred = np.argmax(prediction, axis=1)
#     command = labels[label_pred[0]]
#     print("Predicted label:", command)
#     return command

# if __name__ == "__main__":
#     while True:
#         command = predict_mic()
