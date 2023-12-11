import os
import librosa
import numpy as np
import pandas as pd

import os
import librosa
import pandas as pd
import numpy as np

# 데이터 경로 딕셔너리
data_path_dict = {
    0: ["./dataset/etc/" + file_path for file_path in os.listdir("./dataset/etc/")],
    1: ["./dataset/gaesaekki/" + file_path for file_path in os.listdir("./dataset/gaesaekki/")],
    2: ["./dataset/shibal/" + file_path for file_path in os.listdir("./dataset/shibal/")]
}

# 전체 데이터를 저장할 리스트
all_data = []

# def max len
max_length = 0

# figure max len
for class_label, list_of_files in data_path_dict.items():
    for single_file in list_of_files:
        waveform, sr = librosa.load(single_file, sr=44100)
        max_length = max(max_length, len(waveform))

# add zero-padding
for class_label, list_of_files in data_path_dict.items():
    for single_file in list_of_files:
        waveform, sr = librosa.load(single_file, sr=44100)
        pad_width = max_length - len(waveform)
        waveform = np.pad(waveform, pad_width=(0, pad_width), mode='constant')
        all_data.append([waveform, class_label])
    print(f"Info: Successfully Preprocessed Class Label {class_label}")

# convert df
df = pd.DataFrame(all_data, columns=["feature", "class_label"])

# pkl 파일로 저장
df.to_pickle("final_audio_data_csv/audio_data_waveform.pkl")


# 기존 코드
# def extract_waveform_features(file_path, duration=2, sr=44100):
#     audio, _ = librosa.load(file_path, sr=sr, duration=duration)
#     return audio

# all_data = []
# data_path_dict = {
#     0: ["./myproject/etc/" + file_path for file_path in os.listdir("./myproject/etc/")],
#     1: ["./myproject/gaesaekki/" + file_path for file_path in os.listdir("./myproject/gaesaekki/")],
#     2: ["./myproject/shibal/" + file_path for file_path in os.listdir("./myproject/shibal/")],    
# }

# for class_label, list_of_files in data_path_dict.items():
#     for single_file in list_of_files:
#         waveform = extract_waveform_features(single_file)
#         all_data.append([waveform, class_label])
#     print(f"Info: Successfully Preprocessed Class Label {class_label}")

# df = pd.DataFrame(all_data, columns=["feature", "class_label"])

# # SAVING FOR FUTURE USE
# df.to_pickle("final_audio_data_csv/audio_data_waveform.pkl")
