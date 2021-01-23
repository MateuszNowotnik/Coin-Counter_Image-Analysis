from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Flatten, Dense, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
    ap.add_argument("-m", "--model", required=True, help="path to output model")
    ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output loss/accuracy plot")
    return vars(ap.parse_args())


def baseline_model(resize, depth, classes):
    model = Sequential()

    model.add(Conv2D(20, (5, 5), input_shape=(resize, resize, depth)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(50, (5, 5)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))
    model.add(Dense(classes))
    model.add(Activation("softmax"))

    return model


def main():
    args = get_args()

    # Epochs to train, learning rate and batch size
    EPOCHS = 100
    INIT_LR = 1e-3
    BS = 32
    RESIZE = 28
    DEPTH = 3
    CLASSES = 10

    data = []
    labels = []

    # Grab the image paths
    image_paths = sorted(list(paths.list_images(args["dataset"])))

    # Dictionary of coins
    coins_dict = {'1gr': 0, '2gr': 1, '5gr': 2, '10gr': 3, '20gr': 4,
                  '50gr': 5, '1zl': 6, '2zl': 7, '5zl': 8, 'tail': 9}

    # Loop over the input images
    for image_path in image_paths:
        # Store the pre-processed image in the data list
        image = cv2.imread(image_path)
        image = cv2.resize(image, (RESIZE, RESIZE))
        image = img_to_array(image)
        data.append(image)

        # Extract the class label
        label = image_path.split(os.path.sep)[-2]
        labels.append(coins_dict[label])
    # Scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    # Split the data, 75% for training and 25% for testing
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

    # Convert the labels from integers to vectors
    trainY = to_categorical(trainY, num_classes=CLASSES)
    testY = to_categorical(testY, num_classes=CLASSES)

    # Construct the image generator for data augmentation
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                             horizontal_flip=True, fill_mode="nearest")

    # Initialize the model
    model = baseline_model(RESIZE, DEPTH, CLASSES)
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

    early_stop = callbacks.EarlyStopping(monitor="val_loss",
                                         mode="min", patience=5,
                                         restore_best_weights=True)

    # callbacks=[early_stop]

    # Train the network
    fit = model.fit(x=aug.flow(trainX, trainY, batch_size=BS),
                    validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
                    epochs=EPOCHS, verbose=1)

    # Save the model to disk
    model.save(args["model"], save_format="h5")

    # Plot the graph of loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = len(fit.history['loss'])
    plt.plot(np.arange(0, N), fit.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), fit.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), fit.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), fit.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(args["plot"])


if __name__ == '__main__':
    main()
