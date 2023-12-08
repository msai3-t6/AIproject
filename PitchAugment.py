import os
import librosa
import numpy as np
import soundfile as sf

# 라벨 정의
labels = ["etc", "gaesaekki", "shibal"]

# Pitch variation 함수 정의
def pitch_variation(wav, sample_rate, pitch_factor=2.0):
    # pitch_factor는 피치를 얼마나 바꿀지를 결정하는 계수입니다.
    pitched_wav = librosa.effects.pitch_shift(wav, sr=sample_rate, n_steps=np.random.uniform(-pitch_factor, pitch_factor))
    return pitched_wav

# 각 라벨에 대해 pitch variation을 적용하고 저장
for label in labels:
    # 각 라벨에 해당하는 폴더 생성
    label_folder = f"dataset/{label}_augmentation_pitch"
    os.makedirs(label_folder, exist_ok=True)

    # 라벨 폴더에서 파일 리스트 가져오기
    file_paths = [f"dataset/{label}/" + file_path for file_path in os.listdir(f"dataset/{label}/")]

    # 파일별로 pitch variation 적용 및 저장
    for file_path in file_paths:
        # 음성 파일 로드
        audio, sample_rate = librosa.load(file_path, sr=None)

        # Pitch variation 적용
        audio_pitched = pitch_variation(audio, sample_rate)

        # 저장 경로 설정
        save_path = f"{label_folder}/pitched_{os.path.basename(file_path)}"

        # Pitch variation된 데이터 저장
        sf.write(save_path, audio_pitched, sample_rate)
