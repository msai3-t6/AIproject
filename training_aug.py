####### IMPORTS #############
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from sklearn.metrics import confusion_matrix, classification_report
from plot_cm import plot_confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten

##### Loading saved csv ##############
df = pd.read_pickle("final_audio_data_csv/audio_data.csv")
labels = ["down", "go", "left", "no", "right", "stop", "up", "yes"]
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

# 추가: 어그멘테이션 데이터 불러오기
augmented_df = pd.read_pickle("final_audio_data_csv/augmented_audio_data.csv")
X_augmented = augmented_df["feature"].values
X_augmented = np.concatenate(X_augmented, axis=0).reshape(len(X_augmented), 60)
y_augmented = le.transform(augmented_df["class_label"].tolist())
y_augmented = to_categorical(y_augmented)

# 추가: 기존 데이터와 어그멘테이션 데이터 합치기
X_combined = np.vstack((X_train, X_augmented))
y_combined = np.vstack((y_train, y_augmented))

##### Training ############
# 모델 정의
model = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(X_combined.shape[1], 1)),  # 수정: X_train -> X_combined
    MaxPooling1D(2),
    Conv1D(128, 3, activation='relu'),
    MaxPooling1D(2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(8, activation='softmax') 
])

print(model.summary())

# 모델 컴파일
model.compile(
    loss="categorical_crossentropy",
    optimizer='adam',
    metrics=['accuracy']
)

print("Model Score: \n")
# 모델 학습
history = model.fit(X_combined, y_combined, validation_split=0.2, initial_epoch=0, epochs=1000, callbacks=[es, mc])

# 모델 저장
model.save("saved_model/WWD_vproject.h5")

# 모델 평가
score = model.evaluate(X_test, y_test)
print(score)

#### Evaluating our model ###########
print("Model Classification Report: \n")
y_pred = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
print(classification_report(np.argmax(y_test, axis=1), y_pred))
plot_confusion_matrix(cm, classes=labels)  # labels is the list of your class names