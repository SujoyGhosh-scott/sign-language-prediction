# sign language prediction

in the root of the file we need to add a `/data` folder, that contains all the images in different classes.
in out case there will be 36 subdirs representing a different classes containing images of each class.

to see how the preprocessed images will look, please run `preprocessing-test.py` file. This will run the preprocessing on some of the test images and show the result.

## Data Collection

For testing out the system real time, it is always recommended to collect data instead of using open source datasets.
to Collect data, the `data-collection.py` file will be used.
This script will create a folder called `own-date` containing subdirs of different classes. where image samples will be saved.

- to collect data of a specific class, please update `line 24`, to set the image destination.
  [for example, to collect images of `Z` sign, the value will be `own-data/Z`.]
- make sure to give the class letter in Uppercase in `line 24` like shown in example.
- show the hand sign inside the bordered square of the video frame.
- in the terminal the no of sample images been saved will be shown.
- once desired no of samples are collected, please terminate the process.

## Model Training CNN

to train the model using CNN classifier, please run the `trainCNN.py` file.
this file will read all the images in `all_images` array, preprocess them, and display 20 random preprocessed images.
split the data into train, test, and validation sets, and train the model.
to update the current model please update the `myModel` function.

#### WARNING: do not make changes in the very first Convolution and the very last Dense layer. This will mess up the input and output.

once the model is trained, it shows the accuracy, and saves the model in a pickle file
`model_trained.p` in the root.

### Model Testing

to test the model, please run the `testCNN.py` file.
this file will import the saved model, and make preditions on some of the test images.

## Model Treaining SVM OR RandomForest

to train the model using SVM, please run `trainSVM.py` file.
to train the model using RandomForest, please run `trainRandomForest.py` file.
it works the same way as the CNN model does. but as the model training time depends on the no features and no of samples in SVM, so we had to reduce the image ratio to 16x16.
The trained model is saved in `model_trained_SVM.p` or `model_trained_RandomForest.p` pickle file.

### Model Testing

to test the SVM model, please run `testSVM.py`.
to test the RandomForest model, please run `testRandomForest.py`.
this file will import the saved model, and make preditions on some of the test images.

## Testing from video

to get sign predictions from client video feed, please run the file `realtime-test-ModelType.py`
this script will start get video feed from client device. In the video feed please show the sign in the marked area.
the marked area will be cropped and preprocessed to make prediction. the input feed and the predicted sign will be displayed on the video feed as well.

## Get Confusion-Matrix

To see the confusion matrix for SVM or RandomForest model, please run `confusion-matrix-SVM-RF.py` file.
to choose the model, only the model import section in the beginning needs to be changed.
The rest portion will be same for both.
Before running the file, the images data dir `own-data` has to be present in the root.
