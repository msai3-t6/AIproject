import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

labels = ["down", "go", "left", "no", "right", "stop", "up", "yes"]

all_data = []
df = {}

data_path_dict = {label: [f"{label}/" + file_path for file_path in os.listdir(f"{label}/")] for label in labels}

for class_label, list_of_files in data_path_dict.items():
    for single_file in list_of_files:
        audio, sample_rate = librosa.load(single_file)  ## Loading file

        ##### VISUALIZING WAVE FORM ##
        plt.title(f"Wave Form of {single_file}")
        librosa.display.waveshow(audio, sr=sample_rate)
        # plt.show()

        ##### VISUALIZING MFCC #######
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)  # 60에서 다시 40으로 변경
        print(f"Shape of mfcc for {single_file}:", mfccs.shape)

        plt.title(f"MFCC of {single_file}")
        librosa.display.specshow(mfccs, sr=sample_rate, x_axis='time')
        # plt.show()

        mfcc_processed = np.mean(mfccs.T, axis=0)  ## some pre-processing
        all_data.append([mfcc_processed, class_label])
    print(f"Info: Successfully Preprocessed Class Label {class_label}")

df = pd.DataFrame(all_data, columns=["feature", "class_label"])
print(df)

###### SAVING FOR FUTURE USE ###
df.to_pickle("final_audio_data_csv/audio_data.csv")