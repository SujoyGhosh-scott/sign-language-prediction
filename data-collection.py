import cv2
import numpy as np
import time
import os
import string


# Create the directory structure
if not os.path.exists("own-data"):
    os.makedirs("own-data")
for i in range(9):
    if not os.path.exists("data/" + str(i)):
        os.makedirs("data/"+str(i))

for i in string.ascii_uppercase:
    if not os.path.exists("data/" + i):
        os.makedirs("data/"+i)


cap = cv2.VideoCapture(0)
offset = 20
imgSize = 300
counter = 0
folder = 'own-data/0'


while True:
    _, frame = cap.read()
    ## getting user feed
    cv2.rectangle(frame, (50, 100), (250, 300), (255, 0, 255), 1)

    ## crop from the video feed
    crop = frame[100:300, 50:250]

    # showing the cropped image in the top right of main video frame
    # so we know what we are saving
    bg = np.zeros((frame.shape[0], frame.shape[1]+200, 3), np.uint8)
    bg[:frame.shape[0], :frame.shape[1]] = frame
    bg[:200, frame.shape[1]:frame.shape[1]+200] = crop
    # print(frame.shape, bg.shape, ppd_img.shape)
    cv2.imshow('Frame', bg)

    key = cv2.waitKey(1)
    if key == ord('s'):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', crop)
        print(counter)
