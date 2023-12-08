import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.utils import to_categorical
from plot_cm import plot_confusion_matrix

# #### Loading saved pickle ##############
df = pd.read_pickle("final_audio_data_csv/audio_data_waveform.pkl")

# ####### Making our data training-ready #######
X = np.array(df["feature"].tolist())
X = np.concatenate(X, axis=0).reshape(len(X), X[0].shape[0], 1)  # Reshape for CNN

y = np.array(df["class_label"].tolist())
y = to_categorical(y)

# ####### train test split ############
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ###### Training ############
model = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

print(model.summary())

# 얼리스탑 콜백 정의
early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

# 모델체크포인트 콜백 정의
model_checkpoint = ModelCheckpoint('best_model_waveform.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

# 러닝레이트 스케쥴러 함수 정의
def lr_scheduler(epoch, lr):
    if epoch % 10 == 0:
        return lr * 0.9
    return lr

# 러닝레이트 스케쥴러 콜백 정의
learning_rate_scheduler = LearningRateScheduler(lr_scheduler)

# 모델 컴파일
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

# 훈련 시 fit 함수에 callbacks 매개변수로 추가
history = model.fit(X_train, y_train, epochs=300, callbacks=[early_stopping, model_checkpoint, learning_rate_scheduler], validation_split=0.2, batch_size=32)

# 테스트 데이터로 모델 평가
score = model.evaluate(X_test, y_test)
print("Test Loss and Accuracy: ", score)

# #### Evaluating our model ###########
print("Model Classification Report: \n")
y_pred = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
print(classification_report(np.argmax(y_test, axis=1), y_pred))
plot_confusion_matrix(cm, classes=["etc", "gaesaekki", "shibal"])