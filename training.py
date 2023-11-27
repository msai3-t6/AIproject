import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from sklearn.metrics import confusion_matrix, classification_report

##### Loading saved csv ##############
df = pd.read_pickle("audio_data.csv")

####### Making our data training-ready
X = df["feature"].values
X = np.concatenate(X, axis=0).reshape(len(X), -1)

y = np.array(df["class_label"].tolist())
y = to_categorical(y, num_classes=3)

# MFCC 데이터를 3차원으로 변환
X = np.array([x.reshape(40, 50, 1) for x in X])

####### train test split ############
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

##### Training ############
def model_build():
    model = Sequential()
    
    model.add(Conv2D(128, 3, strides=1, padding='same', activation='relu', input_shape=(40, 50, 1)))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2, padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, 3, strides=1, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2, padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, 3, strides=1, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2, padding='same'))
    model.add(Dropout(0.25))

    model.add(Flatten())
    
    # Adjust the input shape for the dense layer
    model.add(Dense(512, activation='relu', input_shape=(40 * 50,)))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    return model
model = model_build()
model.summary()

# 모델 학습
print("Model Score: \n")
history = model.fit(X_train, y_train, epochs=100)
model.save("saved_model/WWD_ihyun.h5")
score = model.evaluate(X_test, y_test)
print(score)

# #### 모델 평가 ###########
print("Model Classification Report: \n")
y_pred = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
print(classification_report(np.argmax(y_test, axis=1), y_pred))
