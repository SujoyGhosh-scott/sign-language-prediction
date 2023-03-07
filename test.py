import numpy as np
import cv2
import pickle

pickle_in=open("model_trained.p","rb")  ## rb = READ BYTE
model=pickle.load(pickle_in)

all_classes = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
model_inp_img_ratio = 32

model.summary()

local_images_names = ['test_images/1.jpg', 'test_images/5.jpg', 'test_images/I.jpg', 'test_images/W.jpg', 'test_images/X.jpg']
local_images = []

for i in range(len(local_images_names)):
    img = cv2.imread(local_images_names[i])
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img,(5,5),2)
    img = cv2.resize(img, (model_inp_img_ratio, model_inp_img_ratio))
    img = img/255 
    img = img.reshape(1, model_inp_img_ratio, model_inp_img_ratio, 1)

    local_images.append(img)
print('local test images read')


for i in range(len(local_images_names)):
    prediction = model.predict(local_images[i])
    # classIndex = model.predict_classes(local_images[i])
    classIndex = np.argmax(prediction, axis=1)
    probabilityValue =np.amax(prediction)
    print("\nfitting: ", local_images_names[i])
    print("Prediction: ", all_classes[classIndex[0]], " probability: ", probabilityValue)
    # print("predictions: ", prediction)

