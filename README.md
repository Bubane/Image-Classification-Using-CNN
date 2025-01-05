# Image-Classification-Using-CNN
Image Classification using CNN on dataset CIFAR 10

above is direct link for Image Classification Uing CNN
https://colab.research.google.com/drive/1Fk1oyWEt-5a7d3mKaN1IoMnoV2raybRs?usp=sharing

# build an image classification model using convolutional l Networks with TensorFlow or Pytorch use a dataset like CIFAR 10 OR MNIS
# for above we have to import all necessary libraries

# above steps for Image Classification

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, datasets

# now load the dataset CIFAR-10 with TensorFlow

(x_train, y_train),(x_test, y_test) = datasets.cifar10.load_data()

# now normalize the pixel value

x_train, x_test = x_train / 255.0, x_test / 255.0

# Now convert the class vectors into the binary class matrices

y_train= tf.keras.utils.to_categorical(y_train, 10)
y_test= tf.keras.utils.to_categorical(y_test, 10)

# now we can build CNN model 
# convolutional layer 1


model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D(2, 2)
])

# Convolutional Layer 2

Model = models.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2)
])

# Convolutional layer 3

model = models.Sequential([
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2)
])

# Flatten and Fully Connection layer

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Compilation of above model

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#  Train the above model
history=model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test),batch_size=64)

# evaluation of image model

loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy:{accuracy*100:.2}%")

# now plot training and validation accuracy

plt.plot(history.history['accuracy'],
label='Training Accuracy')
plt.show()

plt.plot(history.history['val_accuracy'],
label='Validation Accuracy')
plt.show()

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# plot training and validation losss

plt.plot(history.history['loss'],
label='Training Loss')
plt.show()

plt.plot(history.history['val_loss'],
label='Validation loss')
plt.show()

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
