import os
import cv2
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
# from sklearn.model_selection import train_test_split

### importing saved model

# pickle_in=open("model_trained_RandomForest.p","rb")  # for RandomForest
pickle_in=open("model_trained_SVM.p","rb")  # for SVM
model=pickle.load(pickle_in)

all_classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
# model_inp_img_ratio = 32 # for RandomForest
model_inp_img_ratio = 16 # for SVM

### variables

path = "own-data" # folder with all the class folders
# classes = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


### reading and preprocessing images

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
        # img = img/255            # TO NORMALIZE VALUES BETWEEN 0 AND 1 INSTEAD OF 0 TO 255

        # print("after: ", str(img.shape))

        all_images.append(img.flatten())
        class_no.append(count)
    count+=1

print("all images imported, total images: ", len(all_images))


### forming dataset

all_images = np.array(all_images)
class_no = np.array(class_no)

df = pd.DataFrame(all_images)   # turning into df so each row represents an image and each column represents each pixel value
df['target']=class_no           # adding the output column

print(df.shape)
print(df.head())                # target value should be 0 representing 1
print(df.tail())                # target value should be 34 representing Z

x = df.iloc[:, :-1]             # input data
y = df.iloc[:, -1]              # output data


### getting predictions

y_pred=model.predict(x)


### getting the confusion matrix
cf_matrix = confusion_matrix(y, y_pred)
print('Confusion Matrix')
print(cf_matrix)
sns.heatmap(cf_matrix, annot=True, fmt='.1%', cbar=False, cmap='Blues')
plt.show()
