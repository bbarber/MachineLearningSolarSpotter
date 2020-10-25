
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import os
import tensorboard
from datetime import datetime

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras import backend as K

# classifications notsolar = 0, solar = 1
CLASS_NAMES = ["notsolar", "solar"]
AUTOTUNE = tf.data.experimental.AUTOTUNE
IMG_WIDTH = 256
IMG_HEIGHT = 256

# Use \\ here, if using windows
train_data_dir = '../training_images/*/*.jpg'
test_data_dir = '../test_images/*/*.jpg'
# logdir = '../tensorflow/logs/scalars/' + datetime.now().strftime("%Y%m%d-%H%M%S")

# tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

train_ds = tf.data.Dataset.list_files(train_data_dir)
test_ds = tf.data.Dataset.list_files(test_data_dir)

def get_label(file_path):
  # convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  return parts[-2] == CLASS_NAMES

def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return img

def process_path(file_path):  
  # load the raw data from the file as a string
  label = get_label(file_path)
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label

train_images = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
test_images = test_ds.map(process_path, num_parallel_calls=AUTOTUNE)

model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())

train_images_array = []
train_labels_array = []
test_images_array = []
test_labels_array = []

for img, label in train_images:
  train_images_array.append(img.numpy())
  train_labels_array.append(0 if label.numpy()[0] else 1)

for img, label in test_images:
  test_images_array.append(img.numpy())
  test_labels_array.append(0 if label.numpy()[0] else 1)

train_images_array = np.array(train_images_array)
train_labels_array = np.array(train_labels_array)
test_images_array = np.array(test_images_array)
test_labels_array = np.array(test_labels_array)

# print("train: ")
# print(list(train_labels_array))
# print("test: ")
# print(list(test_labels_array))

model.fit(
  train_images_array, 
  train_labels_array, 
  epochs=1,
  # callbacks=[tensorboard_callback],
)

test_loss, test_acc = model.evaluate(test_images_array,  test_labels_array, verbose=2)

predictions = model.predict(test_images_array)


def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'green'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(CLASS_NAMES[predicted_label],
                                100*np.max(predictions_array),
                                CLASS_NAMES[true_label]),
                                color=color)

def display_predictions():
  num_rows = 5
  num_cols = 3
  num_images = num_rows*num_cols
  plt.figure(figsize=(2*2*num_cols, 2*num_rows))
  for i in range(num_images):
    plt.subplot(num_rows, num_cols, i+1)
    plot_image(i, predictions[i], test_labels_array, test_images_array)
  plt.tight_layout()
  plt.show()

# display_predictions()

# Export the trained model
model.save('dist')

