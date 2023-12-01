import pyaudio
import numpy as np
from tf_helper import preprocess_audiobuffer, get_spectrogram
import tensorflow as tf 
import time
from tensorflow.keras import models

FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16 # 16비트로 설정된 음성 제이터 
CHANNELS = 1
RATE = 16000
p = pyaudio.PyAudio()

def record_audio():
     # PyAudio를 사용하여 스트림을 열고 설정합니다.
    stream = p.open(
        format=FORMAT, #  오디오 포맷을 설정
        channels=CHANNELS, # 오디오 채널 수
        rate=RATE,
        input=True, # True : 마이크로 입력을 처리 
        frames_per_buffer=FRAMES_PER_BUFFER
    )

    print("start recording...")
    #녹음된 오디오 데이터를 저장할 리스트를 초기화합니다.

    frames = []
    # 녹음할 시간을 설정 (1초)
    seconds = 2
    for i in range(0, int(RATE / FRAMES_PER_BUFFER * seconds)): #초당 읽는 데이터 수 * 초 
        data = stream.read(FRAMES_PER_BUFFER)
        frames.append(data)

    # 스트림을 중지하고 닫습니다.
    stream.stop_stream()
    stream.close()
    # 녹음된 오디오 데이터를 NumPy 배열로 변환하여 반환합니다.
    return np.frombuffer(b''.join(frames), dtype=np.int16)




# PyAudio 객체를 종료하는 함수를 정의
def terminate():
    p.terminate()