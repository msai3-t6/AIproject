import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# Function to extract spectrogram features
def extract_spectrogram(file_path, n_fft=2048, hop_length=512):
    audio, sample_rate = librosa.load(file_path)
    spectrogram = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    return log_spectrogram

# Prepare data paths
all_data = []
data_path_dict = {
    0: ["./myproject/etc/" + file_path for file_path in os.listdir("./myproject/etc/")],
    1: ["./myproject/gaesaekki/" + file_path for file_path in os.listdir("./myproject/gaesaekki/")],
    2: ["./myproject/shibal/" + file_path for file_path in os.listdir("./myproject/shibal/")],
}

# Extract spectrogram features and store in DataFrame
for class_label, list_of_files in data_path_dict.items():
    for single_file in list_of_files:
        spectrogram = extract_spectrogram(single_file)
        all_data.append([spectrogram, class_label])
    print(f"Info: Successfully Preprocessed Class Label {class_label}")

df = pd.DataFrame(all_data, columns=["feature", "class_label"])

# Saving for future use
df.to_pickle("final_audio_data_csv/audio_data_spectrogram.pkl")
