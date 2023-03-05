# sign language recognition

in the root of the file we need to add a `/data` folder, that contains all the images in different classes.
in out case there will be 36 subdirs representing a different classes containing images of each class.

to see how the preprocessed images will look, please run `preprocessing-test.py` file. This will run the preprocessing on some of the test images and show the result.

## Model Training

to train the model please run the `train.py` file.
this file will read all the images in `all_images` array, preprocess them, and display 20 random preprocessed images.
split the data into train, test, and validation sets, and train the model.
to update the current model please update the `myModel` function.

### WARNING: do not make changes in the very first Convolution and the very last Dense layer. This will mess up the input and output.

once the model is trained, it shows the accuracy, and saves the model in a pickle file
`model_trained.p` in the root.

## Model Testing

to test the model, please run the `test.py` file.
this file will import the saved model, and make preditions on some of the test images.

## Testing from video

to get sign predictions from client video feed, please run the file `realtime-test.py`
this script will start get video feed from client device. In the video feed please show the sign in the marked area.
the marked area will be cropped and preprocessed to make prediction. the input feed and the predicted sign will be displayed on the video feed as well.
