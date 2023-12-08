import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

# 데이터 로드
df = pd.read_pickle("final_audio_data_csv/audio_data_mel-s.pkl")

# 라벨별로 하나의 데이터를 선택
etc_data = df[df['class_label'] == 0]['feature'].iloc[0]
gaesaekki_data = df[df['class_label'] == 1]['feature'].iloc[0]
shibal_data = df[df['class_label'] == 2]['feature'].iloc[0]

plt.figure(figsize=(5, 5))

# etc 라벨 데이터 시각화
plt.subplot(3, 1, 1)
librosa.display.specshow(etc_data, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('etc')

# gaesaekki 라벨 데이터 시각화
plt.subplot(3, 1, 2)
librosa.display.specshow(gaesaekki_data, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('gaesaekki')

# shibal 라벨 데이터 시각화
plt.subplot(3, 1, 3)
librosa.display.specshow(shibal_data, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('shibal')

plt.tight_layout()
plt.show()

# 라벨별로 Mel Spectrogram의 평균값 계산
etc_mean = np.mean(np.stack(df[df['class_label'] == 0]['feature']), axis=0)
gaesaekki_mean = np.mean(np.stack(df[df['class_label'] == 1]['feature']), axis=0)
shibal_mean = np.mean(np.stack(df[df['class_label'] == 2]['feature']), axis=0)

plt.figure(figsize=(5, 5))

# etc 라벨 데이터의 평균 Mel Spectrogram 시각화
plt.subplot(3, 1, 1)
librosa.display.specshow(etc_mean, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('etc average')

# gaesaekki 라벨 데이터의 평균 Mel Spectrogram 시각화
plt.subplot(3, 1, 2)
librosa.display.specshow(gaesaekki_mean, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('gaesaekki average')

# shibal 라벨 데이터의 평균 Mel Spectrogram 시각화
plt.subplot(3, 1, 3)
librosa.display.specshow(shibal_mean, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('shibal average')

plt.tight_layout()
plt.show()
