import os
import numpy as np
import soundfile as sf
import librosa

labels = ["etc", "gaesaekki", "shibal"]

for label in labels:
    folder_path = f"dataset/{label}/"
    file_paths = [folder_path + file_path for file_path in os.listdir(folder_path) if file_path.endswith('.wav')]

    for file_path in file_paths:
        audio, sample_rate = sf.read(file_path)
        if sample_rate != 44100:
            # if the audio is stereo, resample each channel separately
            if audio.ndim == 2:
                audio_resampled = np.stack([librosa.resample(channel, orig_sr=sample_rate, target_sr=44100) for channel in audio.T])
            else:
                audio_resampled = librosa.resample(audio, orig_sr=sample_rate, target_sr=44100)
            sf.write(file_path, audio_resampled.T, 44100)
            print(f'Resampled {file_path} from {sample_rate} to 44100')