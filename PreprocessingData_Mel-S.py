import os
import librosa
import numpy as np
import pandas as pd

def extract_mel_spectrogram(file_path, n_fft=2048, hop_length=512, n_mels=128):
    audio, sample_rate = librosa.load(file_path, sr=44100)   # Load the audio file
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return log_mel_spectrogram

all_data = []
data_path_dict = {
    0: ["./myproject/etc/" + file_path for file_path in os.listdir("./myproject/etc/")],
    1: ["./myproject/gaesaekki/" + file_path for file_path in os.listdir("./myproject/gaesaekki/")],
    2: ["./myproject/shibal/" + file_path for file_path in os.listdir("./myproject/shibal/")],    
}

for class_label, list_of_files in data_path_dict.items():
    for single_file in list_of_files:
        mel_spectrogram = extract_mel_spectrogram(single_file)
        all_data.append([mel_spectrogram, class_label])
    print(f"Info: Successfully Preprocessed Class Label {class_label}")

df = pd.DataFrame(all_data, columns=["feature", "class_label"])

###### SAVING FOR FUTURE USE ###
df.to_pickle("final_audio_data_csv/audio_data_mel-s.pkl")
