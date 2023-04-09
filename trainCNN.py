import os
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator


#
## defining parameters
#

path = "own-data" # folder with all the class folders
labelFile = 'labels.csv' # file with all names of classes
batch_size_val=50  # how many to process together
steps_per_epoch_val=2000
epochs_val=5
imageDimesions = (128,128,3)
testRatio = 0.2    # if 1000 images split will 200 for testing
validationRatio = 0.2 # if 1000 images 20% of remaining 800 will be 160 for validation
total_classes = 35
# classes = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
model_inp_img_ratio = 32

#
## reading all images
#

count = 0
all_images = [] #stores all the images of all classes
class_no = [] #stores the class index no respect to every image in all images

print("importing all images")

for dir in classes:
    images = os.listdir(path+"/"+dir) #list of all files in a dir
    print("reading images of class: "+dir)
    for img_name in images:
        ###reading the image
        img = cv2.imread(path+"/"+dir+"/"+img_name)

        # print("before preprocessing shape: ", str(img.shape), end=" ")

        ###preprocessing the image
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img,(5,5),2)
        img = cv2.resize(img, (model_inp_img_ratio, model_inp_img_ratio))
        img = img/255            # TO NORMALIZE VALUES BETWEEN 0 AND 1 INSTEAD OF 0 TO 255

        # print("after: ", str(img.shape))

        all_images.append(img)
        class_no.append(count)
    count+=1

print("all images imported, total images: ", len(all_images))
# print(all_images[0].shape)
# print(all_images[0])

### showing 20 random samples of read images
# rand_img_start_idx = 0
# rand_img_end_idx = 2100
# f, axarr = plt.subplots(4,5)
# for i in range(4):
#     for j in range(5):
#         # print(i, j, "between", str((rand_img_start_idx, rand_img_end_idx)), str(random.randrange(rand_img_start_idx, rand_img_end_idx)) ,end=" ")
#         rand_idx = random.randrange(rand_img_start_idx, rand_img_end_idx)
#         axarr[i,j].imshow(all_images[rand_idx], cmap='Greys')
#         axarr[i, j].set_title(classes[class_no[rand_idx]])
#         rand_img_start_idx += 2100
#         rand_img_end_idx += 2100
#     print("")
# plt.show()


#
### splitting data
#
all_images = np.array(all_images)
class_no = np.array(class_no)

x_train, x_test, y_train, y_test = train_test_split(all_images, class_no, test_size=testRatio)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=validationRatio)

# x_train is the array of images to train
# y_train is the corresponding class idx

print("\nData shapes: ")
print("train: ", end="");print(x_train.shape, y_train.shape)
print("validation: ", end="");print(x_validation.shape, y_validation.shape)
print("test: ", end="");print(x_test.shape, y_test.shape)

y_train = to_categorical(y_train,total_classes)
y_validation = to_categorical(y_validation,total_classes)
y_test = to_categorical(y_test,total_classes)


###adding a depth of 1 
#"conv2d" is incompatible with the layer: expected min_ndim=4, found ndim=3. Full shape received: (None, 64, 64)
x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
x_validation=x_validation.reshape(x_validation.shape[0],x_validation.shape[1],x_validation.shape[2],1)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)

#
### CNN model
#
def myModel():
    no_Of_Filters=60
    size_of_Filter=(5,5) # THIS IS THE KERNEL THAT MOVE AROUND THE IMAGE TO GET THE FEATURES.
                         # THIS WOULD REMOVE 2 PIXELS FROM EACH BORDER WHEN USING 32 32 IMAGE
    size_of_Filter2=(3,3)
    size_of_pool=(2,2)  # SCALE DOWN ALL FEATURE MAP TO GERNALIZE MORE, TO REDUCE OVERFITTING
    no_Of_Nodes = 500   # NO. OF NODES IN HIDDEN LAYERS
    model= Sequential()
    model.add((Conv2D(no_Of_Filters,size_of_Filter,input_shape=(model_inp_img_ratio, model_inp_img_ratio, 1),activation='relu')))  # ADDING MORE CONVOLUTION LAYERS = LESS FEATURES BUT CAN CAUSE ACCURACY TO INCREASE
    model.add((Conv2D(no_Of_Filters, size_of_Filter, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool)) # DOES NOT EFFECT THE DEPTH/NO OF FILTERS
 
    model.add((Conv2D(no_Of_Filters//2, size_of_Filter2,activation='relu')))
    model.add((Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(no_Of_Nodes,activation='relu'))
    model.add(Dropout(0.5)) # INPUTS NODES TO DROP WITH EACH UPDATE 1 ALL 0 NONE
    model.add(Dense(total_classes,activation='softmax')) # OUTPUT LAYER
    # COMPILE MODEL
    model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    return model

model = myModel()
print("\nmodel summary")
print(model.summary())

#augmentation of images to make more generic
dataGen= ImageDataGenerator(width_shift_range=0.1,   # 0.1 = 10%     IF MORE THAN 1 E.G 10 THEN IT REFFERS TO NO. OF  PIXELS EG 10 PIXELS
                            height_shift_range=0.1,
                            zoom_range=0.2,  # 0.2 MEANS CAN GO FROM 0.8 TO 1.2
                            shear_range=0.1,  # MAGNITUDE OF SHEAR ANGLE
                            rotation_range=10)  # DEGREES
dataGen.fit(x_train)

#train
history = model.fit_generator(dataGen.flow(x_train, y_train, batch_size=batch_size_val), 
                              #steps_per_epoch=len(x_train), 
                              steps_per_epoch=steps_per_epoch_val, 
                              epochs=epochs_val, 
                              validation_data=(x_validation, y_validation),
                              validation_steps=len(x_test),
                              shuffle=1)

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
score =model.evaluate(x_test,y_test,verbose=0)
print('Test Score:',score[0])
print('Test Accuracy:',score[1])
 
 
# STORE THE MODEL AS A PICKLE OBJECT
pickle_out= open("model_trained.p","wb")  # wb = WRITE BYTE
pickle.dump(model,pickle_out)
pickle_out.close()
print('model saved successfully')
cv2.waitKey(0)