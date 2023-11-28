import os
import shutil

# 변환 대상 폴더의 상위폴더 경로
base_path = "경로"

# 변환 대상 폴더명
target_folders = ['go', 'down', 'left', 'no', 'right', 'stop', 'up', 'yes']

# 각 폴더에 대해 처리
for folder_name in target_folders:
    # 폴더 경로 생성
    folder_path = os.path.join(base_path, folder_name)
    
    # 폴더 내 모든 파일 리스트 가져오기
    file_list = os.listdir(folder_path)
    
    # 파일 정렬 및 숫자 부여를 위한 변수 초기화
    file_list.sort()
    counter = 0
    
    # 변환된 파일명으로 파일 이동
    for file_name in file_list:
        # 파일 확장자 확인 (확장자가 .wav인 파일에 대해서만 작동)
        if file_name.lower().endswith('.wav'):
            # 변환된 파일명 생성
            new_file_name = f'{counter}.wav'
            
            # 파일 이동
            old_path = os.path.join(folder_path, file_name)
            new_path = os.path.join(folder_path, new_file_name)
            shutil.move(old_path, new_path)
            
            # 숫자 증가
            counter += 1

print('모든 폴더의 파일명 변환 완료!')