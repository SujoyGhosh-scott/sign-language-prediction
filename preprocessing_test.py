import cv2
import matplotlib.pyplot as plt

def resize_and_filter(img):
    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img,(5,5),2)
    return img


def preprocessing(img):
    img = resize_and_filter(img)     # CONVERT TO GRAYSCALE

    img = img/255            # TO NORMALIZE VALUES BETWEEN 0 AND 1 INSTEAD OF 0 TO 255
    return img

w_img = cv2.imread('test_images/W.jpg')
ppd_w = preprocessing(w_img)
print("before preprocessing", str(w_img.shape))
print("after preprocessing", str(ppd_w.shape))

one_img = cv2.imread('test_images/1.jpg')
ppd_one = preprocessing(one_img)

f, axarr = plt.subplots(2,2)
axarr[0,0].imshow(w_img)
axarr[0,1].imshow(ppd_w)
axarr[1,0].imshow(one_img)
axarr[1,1].imshow(ppd_one)
plt.show()
