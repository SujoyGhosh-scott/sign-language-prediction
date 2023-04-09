import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Set the paths for the training and validation data
train_dir = 'own-data-preprocessed/train'
val_dir = 'own-data-preprocessed/test'

# Set the image dimensions
img_width, img_height = 32, 32

# Set the number of samples used for each step
train_samples = 1896
val_samples = 400

# Set the number of epochs
epochs = 10

# Set the batch size
batch_size = 32

# Set up the data generators to augment the training images
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# Set up the data generator for the validation images
val_datagen = ImageDataGenerator(rescale=1./255)

# Set up the generator to pull images from the training data folder
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# Set up the generator to pull images from the validation data folder
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# Build the CNN model
model = Sequential()

# Add the first convolutional layer with 32 filters, 3x3 kernel size, and relu activation function
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))

# Add the max pooling layer with 2x2 pool size
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add the second convolutional layer with 64 filters, 3x3 kernel size, and relu activation function
model.add(Conv2D(64, (3, 3), activation='relu'))

# Add the max pooling layer with 2x2 pool size
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add the third convolutional layer with 128 filters, 3x3 kernel size, and relu activation function
model.add(Conv2D(128, (3, 3), activation='relu'))

# Add the max pooling layer with 2x2 pool size
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output from the convolutional layers
model.add(Flatten())

# Add a fully connected layer with 512 units and relu activation function
model.add(Dense(512, activation='relu'))

# Add the output layer with softmax activation function
model.add(Dense(10, activation='softmax'))

# Compile the model with categorical crossentropy loss function, adam optimizer, and accuracy metric
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model using the generators
history = model.fit(
    train_generator,
    steps_per_epoch=train_samples // batch_size,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=val_samples // batch_size)



plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('loss')
plt.xlabel('epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Acurracy')
plt.xlabel('epoch')
plt.show()
 
 
# STORE THE MODEL AS A PICKLE OBJECT
pickle_out= open("model_trained.p","wb")  # wb = WRITE BYTE
pickle.dump(model,pickle_out)
pickle_out.close()
print('model saved successfully')
cv2.waitKey(0)