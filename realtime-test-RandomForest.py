import cv2
# import matplotlib.pyplot as plt
import numpy as np
import pickle

pickle_in=open("model_trained_RandomForest.p","rb")  ## rb = READ BYTE
model=pickle.load(pickle_in)

all_classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
model_inp_img_ratio = 32

camera = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    _, frame = camera.read()
    ## getting user feed
    cv2.rectangle(frame, (50, 100), (250, 300), (255, 0, 255), 1)

    ## crop from the video feed
    crop = frame[100:300, 50:250]
    
    ## preprocess the cropped section
    img = cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img,(5,5),2)
    #img = cv2.flip(img, 1)                  # flipping 180 deg as the input images were taken this way.
    ppd_img = img
    img = cv2.resize(img, (model_inp_img_ratio, model_inp_img_ratio))         # the model was trained with 32x32 images
    
    model_inp = [img.flatten()]

    probabilityValue=model.predict_proba(model_inp)
    classIndex = model.predict(model_inp)[0]
    prediction = all_classes[classIndex]
    print('class idx: ', classIndex, " prediction: ", prediction)
    print('probability')
    print(probabilityValue)

    ## showing prediction in the frame if matches more than 70% of any value
    if(max(probabilityValue[0]) >= 0.7):
        cv2.putText(frame, "class: " + str(classIndex) + " prediction: " + prediction, (80, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    
    ## showing both feed
    # cv2.imshow('Camera', frame)
    # cv2.imshow('Preprocessed', img)

    # adding the preprocessed image in the top right of main video frame
    # sothat we dont have to display them in saperate windows
    bg = np.zeros((frame.shape[0], frame.shape[1]+200, 3), np.uint8)
    bg[:frame.shape[0], :frame.shape[1]] = frame
    bg[:200, frame.shape[1]:frame.shape[1]+200] = ppd_img.reshape(1, 200, 200, 1)
    # print(frame.shape, bg.shape, ppd_img.shape)
    cv2.imshow('Frame', bg)

    ## terminated if x pressed
    if cv2.waitKey(5) == ord('x'):
        break

camera.release()
cv2.destroyAllWindows()