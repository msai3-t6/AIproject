import numpy as np
import tensorflow as tf


# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
def get_spectrogram(waveform):
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(
        waveform, frame_length=1024, frame_step=256)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Transpose the spectrogram to match the shape (height, width, channels).
    spectrogram = tf.transpose(spectrogram, perm=[1, 0])
    # Add a `channels` dimension.
    spectrogram = tf.expand_dims(spectrogram, -1)
    # Resize the spectrogram to the desired shape (688, 129) using bilinear interpolation.
    spectrogram = tf.image.resize(spectrogram, [688, 129], method='bilinear')
    return spectrogram


def preprocess_audiobuffer(waveform):
    """
    waveform: ndarray of size (16000, )
    
    output: Spectogram Tensor of size: (1, `height`, `width`, `channels`)
    """
    #  normalize from [-32768, 32767] to [-1, 1]
    # 32768 :  16비트 PCM(펄스 코드 변조) 인코딩에서 표현 가능한 최대값
    waveform =  waveform / 32768

    waveform = tf.convert_to_tensor(waveform, dtype=tf.float32) # 오디오 웨이브폼을 TensorFlow의 텐서로 변환

    spectogram = get_spectrogram(waveform) # 스펙트로그램을 추출
    
    # add one dimension
    spectogram = tf.expand_dims(spectogram, 0) # 스펙트로그램에 하나의 차원을 추가
    
    return spectogram