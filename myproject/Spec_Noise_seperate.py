import os
import librosa
import numpy as np
import soundfile as sf
import pandas as pd

# 2개 라벨 정의
labels = ["shibal", "gaesaekki"]

def spec_augment(wav):
    D = librosa.stft(wav)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)  # Convert an amplitude spectrogram to dB-scaled spectrogram.

    freq_mask_param = 1  # Hyperparameter for the number of frequency lines to mask
    time_mask_param = 1  # Hyperparameter for the number of time steps to mask

    # Frequency masking
    num_freqs, num_frames = S_db.shape
    freq_mask = np.random.randint(freq_mask_param, size=1)[0] + 1
    f0 = np.random.randint(num_freqs - freq_mask, size=1)[0]

    # Time masking
    time_mask = np.random.randint(time_mask_param, size=1)[0] + 1
    t0 = np.random.randint(num_frames - time_mask, size=1)[0]

    # Apply masks
    S_db_masked = S_db.copy()
    S_db_masked[f0:f0 + freq_mask, :] = 0
    S_db_masked[:, t0:t0 + time_mask] = 0

    return librosa.istft(S_db_masked, length=len(wav))  # Convert the spectrogram back to audio, ensuring the original length


def noise_injection(wav, noise_factor=0.02):
    noise = np.random.randn(len(wav))
    augmented = wav + noise_factor * noise
    augmented = augmented.astype(type(wav[0]))  # Cast back to same data type
    return augmented

def apply_and_save_augmentation(audio, sample_rate, save_path):
    # SpecAugment 적용
    spec_augmented = spec_augment(audio)
    sf.write(save_path + '_spec.wav', spec_augmented, sample_rate)

    # Noise Injection 적용
    noise_injected = noise_injection(audio, noise_factor=0.02)
    sf.write(save_path + '_noise.wav', noise_injected, sample_rate)

# 각 라벨에 대해 augmentation 적용 및 저장
for label in labels:
    # 각 라벨에 해당하는 폴더 생성
    label_folder = f"myproject/dataset/{label}_augmentation"
    os.makedirs(label_folder, exist_ok=True)

    # 라벨 폴더에서 파일 리스트 가져오기
    file_paths = [f"myproject/dataset/{label}/" + file_path for file_path in os.listdir(f"myproject/dataset/{label}/")]

    # 파일별로 augmentation 적용 및 저장
    for file_path in file_paths:
        # 음성 파일 로드
        audio, sample_rate = librosa.load(file_path)

        # 저장 경로 설정
        save_path = f"{label_folder}/augmented_{os.path.basename(file_path).split('.')[0]}"

        # Augmentation 적용 및 저장
        apply_and_save_augmentation(audio, sample_rate, save_path)