import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


#
## defining parameters
#

path = "own-data" # folder with all the class folders
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


# # Calculate the training and testing accuracy scores
# train_accuracy = model.score(x_train, y_train)
# test_accuracy = model.score(x_test, y_test)

# # Store the accuracy scores in two separate lists
# accuracy_scores = [train_accuracy, test_accuracy]
# dataset_labels = ['Training Accuracy', 'Testing Accuracy']

# # Plot the accuracy scores on the same graph
# plt.bar(dataset_labels, accuracy_scores, color=['blue', 'orange'])
# plt.title('SVM Training vs Testing Accuracy')
# plt.xlabel('Dataset')
# plt.ylabel('Accuracy')
# plt.show()

# Get the predicted probabilities for the testing set
probas = model.predict_proba(x_test)

# Calculate the F1 score for each confidence threshold
f1_scores = []
confidence_thresholds = np.linspace(0.0, 1.0, 100)
for threshold in confidence_thresholds:
    y_pred = (probas[:,1] >= threshold).astype(int)
    f1_scores.append(f1_score(y_test, y_pred, average='macro'))

# Plot the F1 score vs confidence threshold
plt.plot(confidence_thresholds, f1_scores)
plt.xlabel('Confidence Threshold')
plt.ylabel('F1 Score')
plt.title('F1 Confidence for SVM')
plt.show()