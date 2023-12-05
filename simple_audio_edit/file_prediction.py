import os
import pathlib
import soundfile as sf
import pandas as pd
import numpy as np
import tensorflow as tf
import glob 
from pathlib import Path
from pydub import AudioSegment
from scipy.io import wavfile
from tensorflow.keras import models



################### 함수 정의 
def get_spectrogram(waveform):
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(
      waveform, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

def change_audio_properties_pred(input_file):
    # 파일 로드
    audio = AudioSegment.from_file(input_file)

    # 모노 채널로 변경
    mono_audio = audio.set_channels(1)

    # 샘플레이트 변경
    resampled_audio = mono_audio.set_frame_rate(16000)

    # 16비트로 변경
    sixteen_bit_audio = resampled_audio.set_sample_width(2)

    # 음성 길이를 2초로 조정
    audio_length = 2000  # 2초
    if len(sixteen_bit_audio) > audio_length:
        # 음성이 2초보다 길면 앞에서부터 잘라내기
        sixteen_bit_audio = sixteen_bit_audio[:audio_length]
    else:
        # 음성이 2초보다 짧으면 뒷부분을 silence로 채우기
        silence = AudioSegment.silent(duration=audio_length - len(sixteen_bit_audio))
        sixteen_bit_audio = sixteen_bit_audio + silence

    # 변환된 오디오 데이터 반환
    return sixteen_bit_audio

def audiosegment_to_raw(audio_segment):
    # Get the raw data
    raw_data = np.array(audio_segment.get_array_of_samples())

    # Convert to float32 (for consistency with the usual audio data)
    raw_data = raw_data.astype(np.float32)

    return raw_data



def predict(model, audio_file_path):
    # 오디오 파일 로드
    audio_segment = change_audio_properties_pred(audio_file_path)

    # AudioSegment 객체를 raw audio data로 변환
    audio = audiosegment_to_raw(audio_segment)
    
    # 오디오 파일을 스펙트로그램으로 변환
    spectrogram = get_spectrogram(audio)

    # 모델의 입력 형태에 맞게 차원을 추가
    spectrogram = np.expand_dims(spectrogram, axis=0)

    # 예측 수행
    prediction = model.predict(spectrogram)

    # 가장 높은 확률을 가진 레이블의 인덱스를 반환
    predicted_index = np.argmax(prediction[0])

    # 레이블 인덱스를 실제 레이블로 변환
    predicted_label = classes[predicted_index]

    return predicted_label


################### 예측 수행 

# 파일 경로 설정 
test_dir = 'test_audio'
test_files = glob.glob(os.path.join(test_dir,'*','*.wav')) 
print(test_files)

# 경로에서 class 추출 
classes = np.array(tf.io.gfile.listdir(str(test_dir)))
print('classes:', classes)
import pandas as pd 

# 결과를 저장할 데이터 프레임 만들기 
file_name = 'result_test.csv'

try:
    df = pd.read_csv(file_name)
except FileNotFoundError:
    df = pd.DataFrame({
        'model': [],
        'filename': [],
        'truelabel': [],
        'predicted_label': [],
        'result': []
    })




model_list = ["saved_model_3class_3800"]
for model_name in model_list: 
    loaded_model = models.load_model(model_name)

    for file in test_files:
        audio = file  
        predicted_label = predict(loaded_model, audio)
        if os.path.basename(os.path.dirname(file)) == predicted_label: 
            result = 'True'
        else:
            result = 'False'

        new_data = {
                    'model' : model_name,
                    'filename': os.path.basename(file), 
                    'truelabel': os.path.basename(os.path.dirname(file)), 
                    'predicted_label': predicted_label, 
                    'result':result}
        df = pd.concat([df, pd.DataFrame([new_data])])
        print("Predicted label:", predicted_label)

    df['result'].value_counts(normalize = True)

    df.to_csv(file_name)


df = pd.read_csv(file_name)

result_df = df.groupby(['model', 'truelabel', 'predicted_label']).agg(
    total=('result', 'count'),
    correct=('result', 'sum')
).reset_index()

result_df['accuracy_percentage'] = (result_df['correct'] / result_df['total'] * 100).astype(str) + '%'

# Group by 'model' and 'truelabel' to get the correct count and accuracy percentage for each true label
agg_result_df = result_df.groupby(['model', 'truelabel']).agg(
    total_correct=('correct', 'sum'),
    total=('total', 'sum')
).reset_index()

agg_result_df['accuracy_percentage'] = round(agg_result_df['total_correct'] / agg_result_df['total'] * 100, 2).astype(str) + '%'

# Print or use agg_result_df as needed
print(agg_result_df)
agg_result_df.to_csv('result_summary.csv')