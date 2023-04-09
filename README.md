# Sign language recognition

Sign language is a visual language that uses a combination of hand gestures, facial expressions, and body movements to convey meaning. It is used primarily by people who are deaf or hard of hearing as a means of communication, but it can also be used by individuals who are not deaf but have difficulty speaking or hearing.

Through this project our goal will be to create a system that will be able to recognise Sign Lanuage from real-time video feed of users with the help of machine learning. There are different types of Sign Languages, some uses gesture to represent words, some expresses letters using hand sings. So, we choose to stick with Indian Sign Language, where different hand signs are used to express different letters, that are used to form sentences.

The project structure is expressed in the following image.

## Data Collection

The First step is Data Collection, where we collect that data on which the machine learning models will be trained. We will soon be adding the link of dataset we have collected. but for now, to test the files, other open source datasets can be used.
For testing out the system real time, it is always recommended to collect data instead of using open source datasets.
to Collect data, the `data-collection.py` file will be used.
This script will create a folder called `own-date` containing subdirs of different classes. where image samples will be saved.

- to collect data of a specific class, please update `line 24`, to set the image destination.
  [for example, to collect images of `Z` sign, the value will be `own-data/Z`.]
- make sure to give the class letter in Uppercase in `line 24` like shown in example.
- show the hand sign inside the bordered square of the video frame.
- in the terminal the no of sample images been saved will be shown.
- once desired no of samples are collected, please terminate the process.

## Data Preprocessing

Preprocessing is an essential step in machine learning because it helps to transform raw data into a format that can be easily understood and processed by machine learning algorithms. In this project, to preprocess the collected images, we first resized the image to 32x32, then turned the image into grayscale and applied Gaussian Filter in the images.

For the SVM and RandomForest model, the preprocessing of the data is done in the training file itself. But for the CNN model, we need to preprocess the collected file saperately. Without this step, the training file (i.e. `/training/trainCNN.py`) won't execute. So if you're only intersted in SVM or RandomForest, this step can be avoided.

To preprocess the collected images, `/proprocessCNN.py` file needs to be executed. This file creates a `own-data-preprocessed` folder in the root. That contains two subdirs `train` and `test`. Both will have all the image classes of `own-data` folder.
In our case, we have around 200 images samples of each classes. and we wanted to to train the model with 80% of the data.
This is why test_samples[`/preprocessCNN.py line 8`] has the value 40, as 20% of 200 images is 40. This many image samples will be stored in the test sample (i.e. in `/own-data-preprocessed/test/[class]`), and the rest in the train sample (i.e. in `/own-data-preprocessed/train/[class]`).
Please change the value accoring to your choice.

## Model Training

All the trained models are stored as pickle file in the root of the project. But to run the training locally, please follow the following steps.

### SVM and Random Forest

The structure of the file and the way they execute are very similar. To train the model using these classifiers, please execute the files`/training/trainSVM.py` or `/training/trainRandomForest.py`. In both of these files, first the data is read, and preprocessed. then the model is trained, and the trained model is saved in the root as a pickle file with the name `model[Classifer].p`.

### CNN

To train the model using CNN classifier, we need to make sure the data is preprocessed, and the `own-data-preprocessed` dir is created properly in the root. Once all these are set, we are ready to train the model using `/training/trainCNN.py` file. This file trains the model, plots the charts of accuracy and data loss wrt training and validation data, and saves the model in the root as a pickle file with the name `modelCNN.p`.

## Model Testing

### Getting Charts

To analize the performance of the trained models please checkout the `/plots` directory. The files in this directory helps to visualise the confusion matrix, f1-score etc to get an idea how efficiently the model works. Before running the files, make sure to have the dataset `own-data` in the root.

### Testing the model real-time

To test the system real-time, please run the `/realtime-test-[Classifier].py` files in the root. This file takes feed from the client device, the user has to show the signs in the marked section, and using the trained model, the signs are recognised to form sentences.

## Application

This is where we create a GUI to make the system usable to non-teachnical people.
