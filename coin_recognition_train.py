from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Flatten, Dense, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


class CoinRecognitionTrain:
    def __init__(self):
        # Epochs to train, learning rate and batch size
        self.EPOCHS = 100
        self.INIT_LR = 1e-3
        self.BS = 32
        self.RESIZE = 28
        self.DEPTH = 3
        self.CLASSES = 10
        self.data = []
        self.labels = []


    def extended_model(self):
        model = Sequential()

        model.add(Conv2D(20, (5, 5), input_shape=(self.RESIZE, self.RESIZE, self.DEPTH)))
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
        model.add(Dense(self.CLASSES))
        model.add(Activation("softmax"))

        return model

    def training_model(self, model, image_paths, model_path):
        # Dictionary of coins
        coins_dict = {'1gr': 0, '2gr': 1, '5gr': 2, '10gr': 3, '20gr': 4,
                      '50gr': 5, '1zl': 6, '2zl': 7, '5zl': 8, 'tail': 9}

        # Loop over the input images
        for image_path in image_paths:
            # Store the pre-processed image in the data list
            image = cv2.imread(image_path)
            image = cv2.resize(image, (self.RESIZE, self.RESIZE))
            image = img_to_array(image)
            self.data.append(image)

            # Extract the class label
            label = image_path.split(os.path.sep)[-2]
            self.labels.append(coins_dict[label])
        # Scale the raw pixel intensities to the range [0, 1]
        data = np.array(self.data, dtype="float") / 255.0
        labels = np.array(self.labels)

        # Split the data, 75% for training and 25% for testing
        (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

        # Convert the labels from integers to vectors
        trainY = to_categorical(trainY, num_classes=self.CLASSES)
        testY = to_categorical(testY, num_classes=self.CLASSES)

        # Image generator for data augmentation
        aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                                 height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                                 horizontal_flip=True, fill_mode="nearest")

        opt = Adam(lr=self.INIT_LR, decay=self.INIT_LR / self.EPOCHS)
        model.compile(loss="categorical_crossentropy", optimizer=opt,
                      metrics=["accuracy"])

        early_stop = callbacks.EarlyStopping(monitor="val_loss",
                                             mode="min", patience=5,
                                             restore_best_weights=True)

        # callbacks=[early_stop]

        # Train the network
        fit = model.fit(x=aug.flow(trainX, trainY, batch_size=self.BS),
                        validation_data=(testX, testY), steps_per_epoch=len(trainX) // self.BS,
                        epochs=self.EPOCHS, verbose=1)

        # Save the model to disk
        model.save(model_path, save_format="h5")

    def plot_graph(self, fit, plot_path):
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
        plt.savefig(plot_path)
