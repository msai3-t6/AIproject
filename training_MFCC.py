import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from plot_cm import plot_confusion_matrix

# 데이터 로드
df = pd.read_pickle("final_audio_data_csv/audio_data_MFCC.pkl")

# feature, class_label 분리
X = np.array(df.feature.tolist())
y = np.array(df.class_label.tolist())

# 클래스 레이블 one-hot encoding
y = to_categorical(y)

# 데이터 차원 확인
num_rows = X[0].shape[0]
num_columns = X[0].shape[1]
num_channels = 1

# train / test 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Conv1D 모델에 맞는 차원으로 변경 (num_rows, num_columns)
X_train = X_train.reshape(X_train.shape[0], num_rows, num_columns)
X_test = X_test.reshape(X_test.shape[0], num_rows, num_columns)

# 모델 구성
model = Sequential()
model.add(Conv1D(filters=16, kernel_size=1, input_shape=(num_rows, num_columns), activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv1D(filters=32, kernel_size=1, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv1D(filters=64, kernel_size=1, activation='relu'))
model.add(Dropout(0.2))

model.add(Conv1D(filters=128, kernel_size=1, activation='relu'))
model.add(Dropout(0.2))
model.add(GlobalAveragePooling1D())

model.add(Dense(y.shape[1], activation='softmax'))

# 모델 컴파일
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# 콜백 설정
callbacks = [ModelCheckpoint(filepath='saved_model/best_model_MFCC.h5', 
                             verbose=1, save_best_only=True),
             EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30),
             ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto', min_lr=0.00001)]

# 모델 학습
history = model.fit(X_train, y_train, batch_size=32, epochs=300, 
                    validation_data=(X_test, y_test), callbacks=callbacks, verbose=1)

# 모델 평가
score = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: ", score[1])

#### Evaluating our model ###########
print("Model Classification Report: \n")
y_pred = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
print(classification_report(np.argmax(y_test, axis=1), y_pred))
plot_confusion_matrix(cm, classes=["etc", "gaesaekki", "shibal"])