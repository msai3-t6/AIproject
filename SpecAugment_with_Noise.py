import os
import librosa
import numpy as np
import soundfile as sf
import pandas as pd

# 라벨 정의
labels = ["etc", "gaesaekki", "shibal"]

# Augmentation 함수 정의
def spec_augment(wav, sample_rate, target_length=2):  # target_length는 원하는 길이로 설정
    n_fft = 2048
    hop_length = 512
    D = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length)  # STFT of y

    freq_mask_param = 2  # Hyperparameter for the number of frequency lines to mask
    time_mask_param = 2  # Hyperparameter for the number of time steps to mask

    # Frequency masking
    num_freqs, num_frames = D.shape
    freq_mask = np.random.randint(freq_mask_param, size=1)[0] + 1
    f0 = np.random.randint(num_freqs - freq_mask, size=1)[0]

    # Time masking
    time_mask = np.random.randint(time_mask_param, size=1)[0] + 1
    t0 = np.random.randint(num_frames - time_mask, size=1)[0]

    # Apply masks
    D_masked = D.copy()
    D_masked[f0:f0 + freq_mask, :] = 0
    D_masked[:, t0:t0 + time_mask] = 0

    # Convert the spectrogram back to audio using Griffin-Lim algorithm
    audio_augmented = librosa.istft(D_masked, hop_length=hop_length)

    # Adjust the length to the target_length
    if len(audio_augmented) < sample_rate * target_length:
        pad_length = sample_rate * target_length - len(audio_augmented)
        audio_augmented = np.pad(audio_augmented, (0, pad_length), 'constant')
    elif len(audio_augmented) > sample_rate * target_length:
        audio_augmented = audio_augmented[:sample_rate * target_length]

    return audio_augmented

# 각 라벨에 대해 augmentation 적용 및 저장

def noise_injection(wav, noise_factor=0.002):
    noise = np.random.uniform(-1, 1, len(wav))
    augmented = wav + noise_factor * noise
    # Cast back to same data type
    augmented = augmented.astype(type(wav[0]))
    return augmented

def gaussian_noise_injection(wav, noise_factor=0.002):
    noise = np.random.randn(len(wav))
    augmented = wav + noise_factor * noise
    # Cast back to same data type
    augmented = augmented.astype(type(wav[0]))
    return augmented

def uniform_noise_injection(wav, noise_factor=0.002):
    noise = np.random.uniform(-1, 1, len(wav))
    augmented = wav + noise_factor * noise
    # Cast back to same data type
    augmented = augmented.astype(type(wav[0]))
    return augmented

def pink_noise_injection(wav, noise_factor=0.002):
    length = len(wav)
    uneven = length % 2
    p = -1.0 # Approximates -1/db per octave

    x = np.r_[2*np.random.randn((length//2)+uneven) + 1j*np.random.randn((length//2)+uneven), 0]
    s = np.sqrt(np.arange(len(x)) + 1.) # Filter
    y = (np.fft.irfft(x/s)).real[:length] # Calculate FFT

    # Normalize the pink noise
    y = y / np.max(np.abs(y))
    # Add the pink noise to the original audio
    augmented = wav + noise_factor * y
    # Cast back to same data type
    augmented = augmented.astype(type(wav[0]))
    return augmented

# 각 라벨에 대해 augmentation 적용 및 저장
for label in labels:
    # 각 라벨에 해당하는 폴더 생성
    label_folder = f"dataset/{label}_augmentation"
    os.makedirs(label_folder, exist_ok=True)

    # 라벨 폴더에서 파일 리스트 가져오기
    file_paths = [f"dataset/{label}/" + file_path for file_path in os.listdir(f"dataset/{label}/")]

    # 파일별로 augmentation 적용 및 저장
    for file_path in file_paths:
        # 음성 파일 로드
        audio, sample_rate = librosa.load(file_path, sr=None)

        # Augmentation 적용
        audio_augmented = spec_augment(audio, sample_rate)
        # 랜덤으로 노이즈 타입 선택
        noise_type = np.random.choice(['gaussian', 'uniform', 'pink'])
        if noise_type == 'gaussian':
            audio_augmented = gaussian_noise_injection(audio_augmented, noise_factor=0.002)
        elif noise_type == 'uniform':
            audio_augmented = uniform_noise_injection(audio_augmented, noise_factor=0.002)
        else:
            audio_augmented = pink_noise_injection(audio_augmented, noise_factor=0.002)

        # 저장 경로 설정
        save_path = f"{label_folder}/augmented_{os.path.basename(file_path)}"

        # Augmented 데이터 저장
        sf.write(save_path, audio_augmented, sample_rate)