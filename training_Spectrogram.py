import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from sklearn.metrics import confusion_matrix, classification_report
from plot_cm import plot_confusion_matrix

# loading saved pickle
df = pd.read_pickle("final_audio_data_csv/audio_data_spectrogram.pkl")

# seperate feature, class_label, convert shape
X = np.array(df["feature"].tolist())
X = np.concatenate(X, axis=0).reshape(len(X), X[0].shape[0], X[0].shape[1], 1)  # Reshape for CNN
y = np.array(df["class_label"].tolist())
y = to_categorical(y)

# train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# def model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

# Model summary
print(model.summary())

# callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model_spectrogram.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

def lr_scheduler(epoch, lr):
    if epoch % 10 == 0:
        return lr * 0.9
    return lr

learning_rate_scheduler = LearningRateScheduler(lr_scheduler)

# Model compilation
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

# Model training
history = model.fit(X_train, y_train, epochs=300, callbacks=[early_stopping, model_checkpoint, learning_rate_scheduler], validation_split=0.2, batch_size=32)

# Model evaluation on test data
score = model.evaluate(X_test, y_test)
print("Test Loss and Accuracy: ", score)

# Evaluating our model
print("Model Classification Report: \n")
y_pred = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
print(classification_report(np.argmax(y_test, axis=1), y_pred))
plot_confusion_matrix(cm, classes=["etc", "gaesaekki", "shibal"])