import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

def extract_spectrogram(file_path):
    y, sr = librosa.load(file_path, sr=44100)
    D = librosa.stft(y)  # STFT of y
    spect = librosa.amplitude_to_db(np.abs(D), ref=np.max)  # Convert to dB scale
    return spect

all_data = []
data_path_dict = {
    0: ["./dataset/etc/" + file_path for file_path in os.listdir("./dataset/etc/")],
    1: ["./dataset/gaesaekki/" + file_path for file_path in os.listdir("./dataset/gaesaekki/")],
    2: ["./dataset/shibal/" + file_path for file_path in os.listdir("./dataset/shibal/")],    
}

# 최대 길이 저장 변수
max_length = 0

# 먼저, 최대 길이를 찾습니다.
for class_label, list_of_files in data_path_dict.items():
    for single_file in list_of_files:
        spect = extract_spectrogram(single_file)
        max_length = max(max_length, spect.shape[1])

# 다시 한번 데이터를 순회하면서, 최대 길이에 맞추어 zero-padding을 추가합니다.
for class_label, list_of_files in data_path_dict.items():
    for single_file in list_of_files:
        spect = extract_spectrogram(single_file)
        pad_width = max_length - spect.shape[1]
        spect = np.pad(spect, pad_width=((0, 0), (0, pad_width)), mode='constant')
        all_data.append([spect, class_label])
    print(f"Info: Successfully Preprocessed Class Label {class_label}")

df = pd.DataFrame(all_data, columns=["feature", "class_label"])

###### SAVING FOR FUTURE USE ###
df.to_pickle("final_audio_data_csv/audio_data_spectrogram.pkl")

# 기존 코드
# # Function to extract spectrogram features
# def extract_spectrogram(file_path, n_fft=2048, hop_length=512):
#     audio, sample_rate = librosa.load(file_path)
#     spectrogram = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
#     log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
#     return log_spectrogram

# # Prepare data paths
# all_data = []
# data_path_dict = {
#     0: ["./dataset/etc/" + file_path for file_path in os.listdir("./dataset/etc/")],
#     1: ["./dataset/gaesaekki/" + file_path for file_path in os.listdir("./dataset/gaesaekki/")],
#     2: ["./dataset/shibal/" + file_path for file_path in os.listdir("./dataset/shibal/")],
# }

# # Extract spectrogram features and store in DataFrame
# for class_label, list_of_files in data_path_dict.items():
#     for single_file in list_of_files:
#         spectrogram = extract_spectrogram(single_file)
#         all_data.append([spectrogram, class_label])
#     print(f"Info: Successfully Preprocessed Class Label {class_label}")

# df = pd.DataFrame(all_data, columns=["feature", "class_label"])

# # Saving for future use
# df.to_pickle("final_audio_data_csv/audio_data_spectrogram.pkl")
