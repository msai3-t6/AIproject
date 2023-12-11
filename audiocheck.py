import os
import soundfile as sf
import librosa

# 라벨별 폴더 경로
folders = ["./dataset/etc/", "./dataset/gaesaekki/", "./dataset/shibal/"]

for folder in folders:
    print(f"Folder: {folder}")
    files = os.listdir(folder)  # 폴더 내의 모든 파일 목록을 가져옴
    files = [file for file in files if file.endswith('.wav')]  # wav 파일만 선택
    
    # 오류가 있는 파일 목록
    error_files = []

    # 상위 5개 파일에 대해
    for file in files[:99999]:
        file_path = os.path.join(folder, file)  # 파일의 전체 경로 생성
        
        try:
            # 파일 로드
            y, sr = librosa.load(file_path, sr=None)  # sr=None으로 설정하면 원래 음성 파일의 샘플링 레이트를 유지

            # 채널 수
            f = sf.SoundFile(file_path)
            channels = f.channels

            # 길이
            duration = librosa.get_duration(y=y, sr=sr)
            
            # 모노/스테레오 판별
            audio_type = 'Mono' if channels == 1 else 'Stereo'

            # 샘플링 레이트
            print(f"File: {file}, Duration: {duration} seconds, Channels: {channels}, Audio Type: {audio_type}, Sampling Rate: {sr}")
        
        except Exception as e:
            # 오류가 발생한 경우 오류 메시지 출력하고 오류가 있는 파일 목록에 추가
            print(f"Error processing file {file}: {str(e)}")
            error_files.append(file)

    # 오류가 있는 파일 목록 출력
    if error_files:
        print(f"\nError files in {folder}: {error_files}")
    else:
        print("No errors found.\n")
