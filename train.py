import argparse
import os

import cv2
import numpy as np
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight


label = {
    'NORMAL': 0,
    'PNEUMONIA': 1
}


def load_images(path):
    X = []
    y = []

    for i in os.listdir(path):
        for j in os.listdir(f'{path}/{i}'):
            image = cv2.imread(f'{path}/{i}/{j}')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (100, 100))
            image = image / 255
            image = image.reshape(100, 100, 1)
            X.append(image)
            y.append(label[i])

    X = np.array(X)
    y = np.array(y)

    return X, y


parser = argparse.ArgumentParser()
parser.add_argument('--data', default='../images')
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--model', default='xray_model.keras')
args = parser.parse_args()

X_train, y_train = load_images(f'{args.data}/train')
X_test, y_test = load_images(f'{args.data}/test')

X_train, y_train = shuffle(X_train, y_train)
X_test, y_test = shuffle(X_test, y_test)

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

model = Sequential()

model.add(Conv2D(filters=100, kernel_size=(5, 5), input_shape=(100, 100, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=100, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=100, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=4, activation='relu'))
model.add(Dense(units=2, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model.fit(
    X_train,
    y_train,
    epochs=args.epochs,
    validation_data=(X_test, y_test),
    class_weight=class_weights,
    callbacks=[early_stop]
)

model.save(args.model)
