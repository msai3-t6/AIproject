import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from plot_cm import plot_confusion_matrix
import numpy as np

##### Loading saved pickle ##############
df = pd.read_pickle("final_audio_data_csv/audio_data_mel-s.pkl")

X = np.array(df.feature.tolist())
y = np.array(df.class_label.tolist())

##### Reshape for CNN input #######
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

###### Train Test Split #######
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

##### CNN Model ########
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(X.shape[1], X.shape[2], 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax')) # 3 classes

model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

##### Callbacks #########
early_stopping = EarlyStopping(monitor='val_loss', patience=30)
model_checkpoint = ModelCheckpoint('saved_model/best_model_mel-s.h5', monitor='val_loss', save_best_only=True)
def scheduler(epoch, lr):
  if epoch < 10:
    return lr
  else:
    return lr * 0.9
lr_scheduler = LearningRateScheduler(scheduler)

##### Training #########
history = model.fit(X_train, y_train, epochs=300, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping, model_checkpoint, lr_scheduler])

##### Evaluating our model ###########
# Test Loss and Accuracy
score = model.evaluate(X_test, y_test)
print("Test Loss and Accuracy: ", score)

# Model Classification Report
y_test_cat = to_categorical(y_test) # 원-핫 인코딩으로 변환
y_pred = np.argmax(model.predict(X_test), axis=1)
print("Model Classification Report: ")
print(classification_report(np.argmax(y_test_cat, axis=1), y_pred))

# Confusion Matrix
cm = confusion_matrix(np.argmax(y_test_cat, axis=1), y_pred)
plot_confusion_matrix(cm, classes=["etc", "gaesaekki", "shibal"])


# ##### Loading saved pickle ##############
# df = pd.read_pickle("final_audio_data_csv/audio_data_mel-s.pkl")

# # ####### Making our data training-ready #######
# X = np.array(df["feature"].tolist())
# X = np.concatenate(X, axis=0).reshape(len(X), X[0].shape[0], X[0].shape[1], 1)  # Reshape for CNN

# y = np.array(df["class_label"].tolist())
# y = to_categorical(y)

# # ####### train test split ############
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # ###### Training ############
# model = Sequential([
#     Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)),
#     MaxPooling2D(pool_size=(2, 2)),
#     Flatten(),
#     Dense(256, activation='relu'),
#     Dropout(0.5),
#     Dense(3, activation='sigmoid') # softmax to sigmoid. softmax 사용시 데이터 confusion matrix 불균형이 매우 심각함. 따라서 sigmoid 변경했으며 훨씬 안정적 분포를 확인
# ])

# print(model.summary())

# # 얼리스탑 콜백 정의
# early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

# # 모델체크포인트 콜백 정의
# model_checkpoint = ModelCheckpoint('best_model_Mel-S.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

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