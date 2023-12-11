import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import random

# 데이터 로드
df = pd.read_pickle("final_audio_data_csv/audio_data_mel-s.pkl")

# 라벨별로 임의의 데이터를 선택
etc_data = df[df['class_label'] == 0]['feature'].iloc[random.randint(0, len(df[df['class_label'] == 0])-1)]
gaesaekki_data = df[df['class_label'] == 1]['feature'].iloc[random.randint(0, len(df[df['class_label'] == 1])-1)]
shibal_data = df[df['class_label'] == 2]['feature'].iloc[random.randint(0, len(df[df['class_label'] == 2])-1)]

plt.figure(figsize=(6, 6))

# etc 라벨 데이터 시각화
plt.subplot(3, 1, 1)
librosa.display.specshow(etc_data, sr=44100, hop_length=1024, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('etc')
plt.xlim([0, 2])  # x축 범위 설정

# gaesaekki 라벨 데이터 시각화
plt.subplot(3, 1, 2)
librosa.display.specshow(gaesaekki_data, sr=44100, hop_length=1024, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('gaesaekki')
plt.xlim([0, 2])  # x축 범위 설정

# shibal 라벨 데이터 시각화
plt.subplot(3, 1, 3)
librosa.display.specshow(shibal_data, sr=44100, hop_length=1024, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('shibal')
plt.xlim([0, 2])  # x축 범위 설정

plt.tight_layout()
plt.show()