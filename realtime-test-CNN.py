import cv2
# import matplotlib.pyplot as plt
import numpy as np
import pickle

pickle_in=open("modelCNN.p","rb")  ## rb = READ BYTE
model=pickle.load(pickle_in)

# all_classes = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
all_classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
model_inp_img_ratio = 32

model.summary()

camera = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

counter = 0
message = ""

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
    img = cv2.resize(img, (32, 32))         # the model was trained with 32x32 images
    img = img/255                           # normalizing values between 0 and 1
    img = np.expand_dims(img, axis=-1)
    img = np.repeat(img, 3, axis=-1)
    img = img.reshape((-1, 32, 32, 3))
    

    #making prediction on the realtime data
    model_prediction = model.predict(img)
    classIndex = np.argmax(model_prediction, axis=1)
    probabilityValue =np.amax(model_prediction)
    prediction = all_classes[classIndex[0]]
    print("Class: ", model_prediction, " probability: ", probabilityValue)

    #showing prediction and generating sentence
    if(probabilityValue>0.95):
        old_letter = prediction
        counter += 1
        print("counter: ", counter)
        if(counter > 50 and prediction == old_letter):
            message += prediction
            old_letter = ""
            counter = 0
        ## showing prediction in the frame
        cv2.putText(frame, "class: " + str(classIndex) + " prediction: " + prediction, (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "message: " + message, (80, 70), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    
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