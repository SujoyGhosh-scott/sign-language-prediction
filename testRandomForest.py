import cv2
import pickle

pickle_in=open("model_trained_RandomForest.p","rb")  ## rb = READ BYTE
model=pickle.load(pickle_in)

all_classes = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
# all_classes = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
model_inp_img_ratio = 32

# model.summary()

local_images_names = ['test_images/1.jpg', 'test_images/5.jpg', 'test_images/I.jpg', 'test_images/W.jpg', 'test_images/X.jpg']
local_images = []

for i in range(len(local_images_names)):
    img = cv2.imread(local_images_names[i])
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img,(5,5),2)
    img = cv2.resize(img, (model_inp_img_ratio, model_inp_img_ratio))

    local_images.append([img.flatten()])
print('local test images read')


for i in range(len(local_images_names)):
    print("\nfitting: ",local_images_names[i])
    # print(local_images[i])
    probability=model.predict_proba(local_images[i])
    for ind,val in enumerate(all_classes):
        print(f'{val} = {round(probability[0][ind]*100, 2)}%, ', end="")
    print("\nThe predicted image is : "+all_classes[model.predict(local_images[i])[0]])

