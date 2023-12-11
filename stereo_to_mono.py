from pydub import AudioSegment
import os

# 변환 함수 정의
def convert_stereo_to_mono(input_folder, output_folder):
    # 출력 폴더가 없으면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 입력 폴더 내 모든 하위 폴더 탐색
    for root, dirs, files in os.walk(input_folder):
        for file_name in files:
            # .wav 파일이 아닌 경우 건너뛰기
            if not file_name.endswith(".wav"):
                continue

            # 입력 파일 경로
            input_path = os.path.join(root, file_name)

            # 새로운 하위 폴더 생성
            relative_path = os.path.relpath(input_path, input_folder)
            output_subfolder = os.path.join(output_folder, os.path.dirname(relative_path))
            if not os.path.exists(output_subfolder):
                os.makedirs(output_subfolder)

            # 출력 파일 경로
            output_path = os.path.join(output_folder, relative_path)

            # 오디오 파일 로드
            audio = AudioSegment.from_wav(input_path)

            # stereo에서 mono로 변환 (channels=1)
            mono_audio = audio.set_channels(1)

            # 변환된 오디오 저장
            mono_audio.export(output_path, format="wav")

# 변환할 폴더 목록
folders_to_convert = ["dataset/etc", "dataset/gaesaekki", "dataset/shibal"]

# 각 폴더에 대해 mono로 변환하여 새로운 폴더에 저장
for folder in folders_to_convert:
    input_folder_path = folder  # 입력 폴더 경로
    output_folder_path = folder + "_mono"  # 출력 폴더 경로

    # stereo를 mono로 변환하여 저장
    convert_stereo_to_mono(input_folder_path, output_folder_path)

print("변환 완료")
