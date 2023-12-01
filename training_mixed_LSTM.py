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
from tensorflow.keras.layers import LSTM # for CNN to LSTM



##### Loading saved csv ##############
df = pd.read_pickle("final_audio_data_csv/audio_data.csv")
labels = ["background", "etc", "gaesaekki", "shibal"]
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

# """
#                                                feature  class_label
# 0    [-479.21857, 120.58049, -18.884195, 24.051443,...            0
# """



####### Making our data training-ready
X = df["feature"].values
X = np.concatenate(X, axis=0).reshape(len(X), 40) #60에서 40으로

# Convert string labels to integers
le = LabelEncoder()
y = le.fit_transform(df["class_label"].tolist())

y = to_categorical(y)

# Expand dimensions
X = np.expand_dims(X, axis=2)

####### train test split ############
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ... (remain the same)

##### Training ############

model = Sequential([ #Conv1D + LSTM
    Conv1D(64, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(2),
    LSTM(128),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax') 
])

print(model.summary())

model.compile(
    loss="categorical_crossentropy",
    optimizer='adam',
    metrics=['accuracy']
)

print("Model Score: \n")
history = model.fit(X_train, y_train, validation_split=0.2, epochs=1000, callbacks=[es, mc])
model.save("saved_model/mixed_lstm.h5")
score = model.evaluate(X_test, y_test)
print(score)

#### Evaluating our model ###########
print("Model Classification Report: \n")
y_pred = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
print(classification_report(np.argmax(y_test, axis=1), y_pred))
plot_confusion_matrix(cm, classes=labels)  # labels is the list of your class names
