import os
import librosa
import numpy as np
import soundfile as sf
import pandas as pd

# 8개 라벨 정의
labels = ["down", "go", "left", "no", "right", "stop", "up", "yes"]

# Augmentation 함수 정의
def apply_augmentation(audio, sample_rate):
    # Pitch shift
    pitch_shifted = librosa.effects.pitch_shift(audio, n_steps=2, sr=sample_rate)

    # Time stretch
    time_stretched = librosa.effects.time_stretch(audio, rate=1.2)

    # 배열을 조인
    augmented_audio = np.concatenate([pitch_shifted, time_stretched])

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


# 최종 어그멘테이션된 데이터를 CSV 파일로 저장
augmented_df.to_csv("final_audio_data_csv/augmented_audio_data.csv", index=False)