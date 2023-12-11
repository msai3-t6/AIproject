import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# data path
data_path_dict = {
    0: ["./dataset/etc/" + file_path for file_path in os.listdir("./dataset/etc/")],
    1: ["./dataset/gaesaekki/" + file_path for file_path in os.listdir("./dataset/gaesaekki/")],
    2: ["./dataset/shibal/" + file_path for file_path in os.listdir("./dataset/shibal/")],    
}

# save list
all_data = []

# def max length
max_length = 0

# figure max len
for class_label, list_of_files in data_path_dict.items():
    for single_file in list_of_files:
        audio, sample_rate = librosa.load(single_file, sr=44100) ## Loading file
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        max_length = max(max_length, mfcc.shape[1])

# add zero-padding
for class_label, list_of_files in data_path_dict.items():
    for single_file in list_of_files:
        audio, sample_rate = librosa.load(single_file, sr=44100)
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        pad_width = max_length - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        all_data.append([mfcc, class_label])
    print(f"Info: Successfully Preprocessed Class Label {class_label}")

# convert df
df = pd.DataFrame(all_data, columns=["feature", "class_label"])

# pkl 파일로 저장
df.to_pickle("final_audio_data_csv/audio_data_MFCC.pkl")
