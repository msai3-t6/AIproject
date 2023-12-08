import os
import librosa
import numpy as np
import pandas as pd

def extract_waveform_features(file_path, duration=2, sr=44100):
    audio, _ = librosa.load(file_path, sr=sr, duration=duration)
    return audio

all_data = []
data_path_dict = {
    0: ["./myproject/etc/" + file_path for file_path in os.listdir("./myproject/etc/")],
    1: ["./myproject/gaesaekki/" + file_path for file_path in os.listdir("./myproject/gaesaekki/")],
    2: ["./myproject/shibal/" + file_path for file_path in os.listdir("./myproject/shibal/")],    
}

for class_label, list_of_files in data_path_dict.items():
    for single_file in list_of_files:
        waveform = extract_waveform_features(single_file)
        all_data.append([waveform, class_label])
    print(f"Info: Successfully Preprocessed Class Label {class_label}")

df = pd.DataFrame(all_data, columns=["feature", "class_label"])

# SAVING FOR FUTURE USE
df.to_pickle("final_audio_data_csv/audio_data_waveform.pkl")
