import argparse

import cv2
import numpy as np
from keras.models import load_model


classes = {
    0: 'NORMAL',
    1: 'PNEUMONIA'
}


parser = argparse.ArgumentParser()
parser.add_argument('--image', required=True)
parser.add_argument('--model', default='xray_model.keras')
args = parser.parse_args()

model = load_model(args.model)

image = cv2.imread(args.image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.resize(image, (100, 100))
image = image / 255
image = image.reshape(1, 100, 100, 1)

predict = model.predict(image)
answer = np.argmax(predict)

print(classes[answer])
print(predict)
