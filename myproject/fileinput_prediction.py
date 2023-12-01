# audio_file_prediction.py
import librosa
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

base_path = './myproject/dataset/'

# Update your class names
class_names = ["etc", "gaesaekki", "shibal"]

# Loading our saved model
model = load_model("saved_model/myproject_modified_cnn.h5")

# Specify your audio file path
file_path = base_path + "etc/90.wav"

# Load your audio file
audio, sample_rate = librosa.load(file_path)

# Extract MFCC features
mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
mfcc_processed = np.mean(mfcc.T, axis=0)

# Make predictions
prediction = model.predict(np.expand_dims(mfcc_processed, axis=0))
predicted_index = np.argmax(prediction)
print(f"Predicted class: {class_names[predicted_index]}")