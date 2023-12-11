import os
import librosa
import numpy as np
import pandas as pd

def extract_mel_spectrogram(file_path):
    y, sr = librosa.load(file_path, sr=44100)
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=1024)
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    return mel_spect

all_data = []
data_path_dict = {
    0: ["./dataset/etc/" + file_path for file_path in os.listdir("./dataset/etc/")],
    1: ["./dataset/gaesaekki/" + file_path for file_path in os.listdir("./dataset/gaesaekki/")],
    2: ["./dataset/shibal/" + file_path for file_path in os.listdir("./dataset/shibal/")],    
}

# def max length
max_length = 0

# figure max len
for class_label, list_of_files in data_path_dict.items():
    for single_file in list_of_files:
        mel_spect = extract_mel_spectrogram(single_file)
        max_length = max(max_length, mel_spect.shape[1])

# add zero-padding
for class_label, list_of_files in data_path_dict.items():
    for single_file in list_of_files:
        mel_spect = extract_mel_spectrogram(single_file)
        pad_width = max_length - mel_spect.shape[1]
        mel_spect = np.pad(mel_spect, pad_width=((0, 0), (0, pad_width)), mode='constant')
        all_data.append([mel_spect, class_label])
    print(f"Info: Successfully Preprocessed Class Label {class_label}")

# convert df
df = pd.DataFrame(all_data, columns=["feature", "class_label"])

###### SAVING FOR FUTURE USE ###
df.to_pickle("final_audio_data_csv/audio_data_mel-s.pkl")

# 216 프레임 padding or truncating 방식

# def extract_mel_spectrogram(file_path):
#     y, sr = librosa.load(file_path, sr=44100)
#     mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=1024)
#     mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    
#     # Padding or truncating to have consistent shape
#     if mel_spect.shape[1] > 216:
#         mel_spect = mel_spect[:,:216]
#     else:
#         mel_spect = np.pad(mel_spect, ((0,0), (0, 216 - mel_spect.shape[1])))
    
#     return mel_spect

# all_data = []
# data_path_dict = {
#     0: ["./dataset/etc/" + file_path for file_path in os.listdir("./dataset/etc/")],
#     1: ["./dataset/gaesaekki/" + file_path for file_path in os.listdir("./dataset/gaesaekki/")],
#     2: ["./dataset/shibal/" + file_path for file_path in os.listdir("./dataset/shibal/")],    
# }

# for class_label, list_of_files in data_path_dict.items():
#     for single_file in list_of_files:
#         mel_spectrogram = extract_mel_spectrogram(single_file)
#         all_data.append([mel_spectrogram, class_label])
#     print(f"Info: Successfully Preprocessed Class Label {class_label}")

# df = pd.DataFrame(all_data, columns=["feature", "class_label"])

# ###### SAVING FOR FUTURE USE ###
# df.to_pickle("final_audio_data_csv/audio_data_mel-s.pkl")