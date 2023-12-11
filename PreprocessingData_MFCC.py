import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 데이터 경로 딕셔너리
data_path_dict = {
    0: ["./dataset/etc/" + file_path for file_path in os.listdir("./dataset/etc/")],
    1: ["./dataset/gaesaekki/" + file_path for file_path in os.listdir("./dataset/gaesaekki/")],
    2: ["./dataset/shibal/" + file_path for file_path in os.listdir("./dataset/shibal/")],    
}

# 전체 데이터를 저장할 리스트
all_data = []

# 최대 길이 저장 변수
max_length = 0

# 먼저, 최대 길이를 찾습니다.
for class_label, list_of_files in data_path_dict.items():
    for single_file in list_of_files:
        audio, sample_rate = librosa.load(single_file, sr=44100) ## Loading file
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        max_length = max(max_length, mfcc.shape[1])

# 다시 한번 데이터를 순회하면서, 최대 길이에 맞추어 zero-padding을 추가합니다.
for class_label, list_of_files in data_path_dict.items():
    for single_file in list_of_files:
        audio, sample_rate = librosa.load(single_file, sr=44100)
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        pad_width = max_length - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        all_data.append([mfcc, class_label])
    print(f"Info: Successfully Preprocessed Class Label {class_label}")

# DataFrame으로 변환
df = pd.DataFrame(all_data, columns=["feature", "class_label"])

# pkl 파일로 저장
df.to_pickle("final_audio_data_csv/audio_data_MFCC.pkl")
