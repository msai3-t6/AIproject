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

labels = ["etc", "gaesaekki", "shibal"]

# Loading saved csv
df = pd.read_pickle("final_audio_data_csv/audio_data.csv")

# Making our data training-ready
X = df["feature"].values
X = np.concatenate(X, axis=0).reshape(len(X), 40)
print(X)

y = np.array(df["class_label"].tolist())
y = to_categorical(y)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1) # 3차원 변경
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1) # 3차원 변경

# Training
model = Sequential()
model.add(Conv1D(64, 3, activation='relu', input_shape=X_train[0].shape))
model.add(MaxPooling1D(2))
model.add(Conv1D(128, 3, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax')) # Assuming you have 4 labels

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
history = model.fit(X_train, y_train, epochs=300, callbacks=[es, mc], validation_split=0.2)

print("Model Score: \n")
history = model.fit(X_train, y_train, epochs=300)
model.save("saved_model/myproject_modified_cnn.h5")
score = model.evaluate(X_test, y_test)
print(score)

# Evaluating our model #
print("Model Classification Report: \n")
y_pred = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
print(classification_report(np.argmax(y_test, axis=1), y_pred))
plot_confusion_matrix(cm, classes=labels)  # labels is the list of your class names