from pydub import AudioSegment
import os

# 현재 작업 디렉토리 설정 (스크립트 파일이나 노트북 파일의 위치에 따라 적절히 수정)
current_directory = os.getcwd()
originvoice_directory = os.path.join(current_directory, "originvoice")

# originvoice 폴더 안에 있는 모든 파일을 가져오기
audio_files = [f for f in os.listdir(originvoice_directory) if f.endswith(".wav")]

for audio_file in audio_files:
    # 음성 파일 로드
    audio = AudioSegment.from_file(os.path.join(originvoice_directory, audio_file), format="wav")

    # 2초를 밀리세컨드로 변환 (pydub에서 시간은 밀리세컨드 단위로 측정)
    two_seconds = 2 * 1000

    # 분할 시작
    chunks = [audio[i:i+two_seconds] for i in range(0, len(audio), two_seconds)]

    # 각 조각을 별도의 파일로 저장
    for i, chunk in enumerate(chunks):
        output_file_path = os.path.join(originvoice_directory, f"chunk_{audio_file}_{i}.wav")
        chunk.export(output_file_path, format="wav")
        print(f"Saved: {output_file_path}")
