import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from plot_cm import plot_confusion_matrix

# loading saved pickle
df = pd.read_pickle("final_audio_data_csv/audio_data_waveform.pkl")

# seperate feature, class_label, convert shape
X = np.array(df.feature.tolist())
y = np.array(df.class_label.tolist())

# one-hot encoding
y = to_categorical(y)

# check shape
num_rows = X[0].shape[0]

# train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# convert shape
X_train = X_train.reshape(X_train.shape[0], num_rows, 1)
X_test = X_test.reshape(X_test.shape[0], num_rows, 1)

# model
model = Sequential()
model.add(Conv1D(filters=16, kernel_size=2, input_shape=(num_rows, 1), activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv1D(filters=32, kernel_size=2, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
model.add(Dropout(0.2))

model.add(Conv1D(filters=128, kernel_size=2, activation='relu'))
model.add(Dropout(0.2))
model.add(GlobalAveragePooling1D())

model.add(Dense(y.shape[1], activation='softmax'))

# compile
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# callbacks
callbacks = [ModelCheckpoint(filepath='saved_model/best_model_waveform.h5', verbose=1, save_best_only=True),
             EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30),
             ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto', min_lr=0.00001)]

# Model training
history = model.fit(X_train, y_train, batch_size=32, epochs=300, validation_data=(X_test, y_test), callbacks=callbacks, verbose=1)

# Model evaluation on test data
score = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: ", score[1])

# Evaluating our model
print("Model Classification Report: ")
y_pred = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
print(classification_report(np.argmax(y_test, axis=1), y_pred))
plot_confusion_matrix(cm, classes=["etc", "gaesaekki", "shibal"])


# 기존 코드
# # #### Loading saved pickle ##############
# df = pd.read_pickle("final_audio_data_csv/audio_data_waveform.pkl")

# # ####### Making our data training-ready #######
# X = np.array(df["feature"].tolist())
# X = np.concatenate(X, axis=0).reshape(len(X), X[0].shape[0], 1)  # Reshape for CNN

# y = np.array(df["class_label"].tolist())
# y = to_categorical(y)

# # ####### train test split ############
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # ###### Training ############
# model = Sequential([
#     Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
#     MaxPooling1D(pool_size=2),
#     Flatten(),
#     Dense(256, activation='relu'),
#     Dropout(0.5),
#     Dense(3, activation='softmax')
# ])

# print(model.summary())

# # 얼리스탑 콜백 정의
# early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

# # 모델체크포인트 콜백 정의
# model_checkpoint = ModelCheckpoint('best_model_waveform.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

# # 러닝레이트 스케쥴러 함수 정의
# def lr_scheduler(epoch, lr):
#     if epoch % 10 == 0:
#         return lr * 0.9
#     return lr

# # 러닝레이트 스케쥴러 콜백 정의
# learning_rate_scheduler = LearningRateScheduler(lr_scheduler)

# # 모델 컴파일
# model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

# # 훈련 시 fit 함수에 callbacks 매개변수로 추가
# history = model.fit(X_train, y_train, epochs=300, callbacks=[early_stopping, model_checkpoint, learning_rate_scheduler], validation_split=0.2, batch_size=32)

# # 테스트 데이터로 모델 평가
# score = model.evaluate(X_test, y_test)
# print("Test Loss and Accuracy: ", score)

# # #### Evaluating our model ###########
# print("Model Classification Report: \n")
# y_pred = np.argmax(model.predict(X_test), axis=1)
# cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
# print(classification_report(np.argmax(y_test, axis=1), y_pred))
# plot_confusion_matrix(cm, classes=["etc", "gaesaekki", "shibal"])