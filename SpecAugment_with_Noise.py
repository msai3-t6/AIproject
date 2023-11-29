import os
import librosa
import numpy as np
import soundfile as sf
import pandas as pd

# 8개 라벨 정의
labels = ["down", "go", "left", "no", "right", "stop", "up", "yes"]

# Augmentation 함수 정의
def spec_augment(wav):
    D = librosa.stft(wav)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)  # Convert an amplitude spectrogram to dB-scaled spectrogram.

    freq_mask_param = 10  # Hyperparameter for the number of frequency lines to mask
    time_mask_param = 10  # Hyperparameter for the number of time steps to mask

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

    return librosa.istft(S_db_masked)  # Convert the spectrogram back to audio

def noise_injection(wav, noise_factor=0.02):
    noise = np.random.randn(len(wav))
    augmented = wav + noise_factor * noise
    # Cast back to same data type
    augmented = augmented.astype(type(wav[0]))
    return augmented

def apply_augmentation(audio, sample_rate):
    # SpecAugment
    spec_augmented = spec_augment(audio)

    # Noise Injection
    noise_injected = noise_injection(audio, noise_factor=0.02)

    # 배열을 조인
    augmented_audio = np.concatenate([spec_augmented, noise_injected])
    return augmented_audio


# 라벨에 대한 어그멘테이션 데이터를 저장할 데이터프레임 생성
augmented_df = pd.DataFrame(columns=["feature", "class_label"])

# 각 라벨에 대해 augmentation 적용 및 저장
for label in labels:
    # 각 라벨에 해당하는 폴더 생성
    label_folder = f"{label}_augmentation"
    os.makedirs(label_folder, exist_ok=True)

    # 라벨 폴더에서 파일 리스트 가져오기
    file_paths = [f"{label}/" + file_path for file_path in os.listdir(f"{label}/")]

    # 파일별로 augmentation 적용 및 저장
    for file_path in file_paths:
        # 음성 파일 로드
        audio, sample_rate = librosa.load(file_path)

        # Augmentation 적용
        augmented_audio = apply_augmentation(audio, sample_rate)

        # 저장 경로 설정
        save_path = f"{label_folder}/augmented_{os.path.basename(file_path)}"

        # Augmented 데이터 저장
        sf.write(save_path, augmented_audio, sample_rate)

        # 데이터프레임에 정보 추가
        df = pd.DataFrame({"feature": [augmented_audio], "class_label": [label]})
        augmented_df = augmented_df._append(df, ignore_index=True)