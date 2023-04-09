import os
import cv2
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


#
## defining parameters
#

root = os.getcwd() # folder with all the class folders
print("project root: ", root)
# classes = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
model_inp_img_ratio = 16

#
## reading all images
#

count = 0
all_images = [] #stores all the images of all classes
class_no = [] #stores the class index no respect to every image in all images

print("importing all images")

for dir in classes:
    images = os.listdir(root + '/own-data/' + dir)
    print("reading images of class: "+dir)
    for img_name in images:
        ###reading the image
        img = cv2.imread(root + '/own-data/' + dir + "/" + img_name)

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
# print(all_images[0].shape)
# print(all_images[0])


#
### forming dataset
#
all_images = np.array(all_images)
class_no = np.array(class_no)

df = pd.DataFrame(all_images)   # turning into df so each row represents an image and each column represents each pixel value
df['target']=class_no           # adding the output column
print(df.shape)
print(df.head())                # target value should be 0 representing 1
print(df.tail())                # target value should be 34 representing Z


# splitting dataset

x = df.iloc[:, :-1]             # input data
y = df.iloc[:, -1]              # output data

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=77,stratify=y)
print('Splitted Successfully')

#Training

param_grid = {'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf','poly']}
svc = svm.SVC(probability=True)
model = GridSearchCV(svc,param_grid)

model.fit(x_train,y_train)
print('The Model is trained well with the given images')
# model.best_params_ contains the best parameters obtained from GridSearchCV


y_pred=model.predict(x_test)
# print("The predicted Data is :")
# print(y_pred)
# print("The actual data is:")
# print(np.array(y_test))
print(f"The model is {accuracy_score(y_pred,y_test)*100}% accurate")

# STORE THE MODEL AS A PICKLE OBJECT
pickle_path = os.path.join(root, "modelSVM.p")
pickle_out= open(pickle_path,"wb")  # wb = WRITE BYTE
pickle.dump(model,pickle_out)
pickle_out.close()
print('model saved successfully')
cv2.waitKey(0)
