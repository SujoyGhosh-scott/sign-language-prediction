import cv2
import os

data_folder = 'own-data'
output_folder = 'own-data-preprocessed'
output_img_dim = 32 # the output image will be 32x32
total_train_images = 0

# Create the output folder if it does not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
# creating train, test subfolders in the output folder
if not os.path.exists(output_folder + '/train'):
    os.makedirs(output_folder + '/train')
if not os.path.exists(output_folder + '/test'):
    os.makedirs(output_folder + '/test')

# Loop over all subfolders in the data folder
for folder_name in os.listdir(data_folder):
    folder_path = os.path.join(data_folder, folder_name)
    if os.path.isdir(folder_path):
        # Loop over all files in the subfolder
        print('preprocessing: ', folder_name)
        count = 0
        mode = 'test'
        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)
            # Load the image
            image = cv2.imread(image_path)
            # print(image.shape)

            ## preprocess the image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.GaussianBlur(image,(5,5),2)
            image = cv2.resize(image, (output_img_dim, output_img_dim))

            ## create subdir if not already exist in the subfolder in the output folder
            subdir_path = output_folder + '/' + mode + '/' + folder_name
            if not os.path.exists(subdir_path):
                os.makedirs(subdir_path)

            ## save the image in subfolder
            ppd_img_name = filename + '.jpg'
            ppd_img_path = subdir_path + '/' + ppd_img_name
            cv2.imwrite(ppd_img_path, image)

            count += 1
            if(count == 39):
                mode = 'train'
            if(count > 39):
                total_train_images += 1

print('total train images: ', total_train_images)
