import os 
import glob
from pydub import AudioSegment
import numpy as np 
import pandas as pd
import librosa
from sklearn.preprocessing import MinMaxScaler
import soundfile as sf



# pydub : 오디오 관리 모듈 
# pydub 사용을 위해 ffmpeg 설치 (참고링크 https://digital-play.tistory.com/104) 

#wav 파일로 변형하는 함수 정의 
def convert_to_wav(file_path, number):
    audio = AudioSegment.from_file(file_path) # 기존 음성을 불러오기 
    directory = os.path.dirname(file_path)
    wav_file_name = os.path.join(directory.replace('dataset', 'wav_dataset'), number) + '.wav'
    audio.export(wav_file_name, format='wav') # wav_file_name 에 wav format 으로 내보내기 
    return wav_file_name

def remove_silence_with_librosa(filename, top_db = 30):
    # 오디오 파일 읽기
    y, sr = librosa.load(filename, sr=None)

    # 무음 부분 제거
    intervals = librosa.effects.split(y, top_db=top_db)

    # 음성이 있는 부분만을 남김
    non_silent_audio = np.concatenate([y[start:end] for start, end in intervals])

    # 음성이 있는 부분만을 포함하는 오디오 파일 생성
    output_filename = filename.replace('wav_dataset', 'remove_dataset')
    output_filename = output_filename.replace('.wav', '_no_silence.wav')

    sf.write(output_filename, non_silent_audio, sr)

    return output_filename




# 라벨별 데이터 셋 생성 
labels = os.listdir('dataset') 
for label in labels: 
    os.makedirs('wav_dataset/'+label, exist_ok=True)
    os.makedirs('remove_dataset/'+label, exist_ok=True)
    

# 데이터 셋 리스트 만들기
org_path = 'dataset'
org_audio_list = glob.glob(os.path.join(org_path, '*', "*"))

# 데이터 셋 순회하며 wav 파일로 변환 
for idx, audio in enumerate(org_audio_list): 
    convert_to_wav(audio, str(idx))

# 음성 무음구간(앞뒤) 삭제
wav_path = 'wav_dataset'
org_wav_list = glob.glob(os.path.join(wav_path, '*', "*"))
for wav in org_wav_list: 
    remove_silence_with_librosa(wav)


all_data = []
df = {}

data_path_dict = {
    0: ["negative_audio/" + file_path for file_path in os.listdir("negative_audio/")],
    1: ["remove_dataset/gaesaekki/" + file_path for file_path in os.listdir("remove_dataset/gaesaekki/")],
    2: ["remove_dataset/shibal/" + file_path for file_path in os.listdir("remove_dataset/shibal/")]
}

for class_label, list_of_files in data_path_dict.items():
    for single_file in list_of_files:
        audio, sample_rate = librosa.load(single_file) ## Loading file
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40) ## Apllying mfcc
        mfcc_np = np.array(mfcc, np.float32)
        all_data.append([mfcc_np, class_label])
        print(single_file, mfcc_np.shape)
        
    print(f"Info: Succesfully Preprocessed Class Label {class_label}")


max_length = 50  # 모든 MFCC가 이 길이가 되도록 조정

for i in range(len(all_data)):
    mfcc = all_data[i][0]  # i번째 샘플의 MFCC
    if mfcc.shape[1] < max_length:  # mfcc의 길이가 max_length보다 짧은 경우
        mfcc_fixed_length = np.pad(mfcc, ((0, 0), (0, max_length - mfcc.shape[1])), mode='constant')
    else:
        mfcc_fixed_length = mfcc[:, :max_length]  # mfcc의 길이가 max_length보다 긴 경우, 처음부터 max_length까지 잘라냄
    all_data[i][0] = mfcc_fixed_length

df = pd.DataFrame(all_data, columns=["feature", "class_label"])


###### SAVING FOR FUTURE USE ###
df.to_pickle("audio_data.csv")
 # 파이썬 객체를 이진 형식으로 저장
