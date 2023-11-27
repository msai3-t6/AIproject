import pyaudio
import numpy as np
import scipy.io.wavfile as wav
from scipy.fftpack import fft
from tensorflow.keras.models import load_model
import librosa

# PyAudio Configuration
CHUNK_SIZE = 22050*2  # 1 second of audio at 22050 Hz
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 20000

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Start the stream
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK_SIZE)


model = load_model("saved_model/WWD_ihyun.h5")
classes = ["nothing", "gaesaeki", "shibal"]

while True:
    # Read the data from the stream
    data = stream.read(CHUNK_SIZE)
    data = np.frombuffer(data, dtype=np.int16)
    data_float32 = data.astype(np.float32)
    
    # Convert to MFCC
    mfcc = librosa.feature.mfcc(y=np.expand_dims(data_float32, axis=0), sr=RATE, n_mfcc=40)

    # Make sure the audio input is the right shape
    if mfcc.shape[2] < 50:
        pad_width = 50 - mfcc.shape[2]
        mfcc = np.pad(mfcc, ((0, 0), (0, 0), (0, pad_width)), mode='constant', constant_values=0)
    else:
        mfcc = mfcc[:, :, :50]

    mfcc = mfcc.reshape(1, 40, 50, 1)

    # Predict the class of the audio input
    prediction = model.predict(mfcc)
    predicted_class = np.argmax(prediction)
    print('Prediction: ', classes[predicted_class])


# Stop the stream
stream.stop_stream()
stream.close()

# Close PyAudio
audio.terminate()
