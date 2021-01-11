import numpy as np
import os
import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras.optimizers import *
from keras import regularizers


def get_train_test(train_ratio, dataPath):

    labels, _, _ = get_labels(dataPath)
    classNumber = 0

    # On init avec le premier tableau pour avoir les bonnes dimensions pour la suite
    X = data = np.load(dataPath + '\\' + labels[0] + '.npy')
    Y = np.zeros(X.shape[0])
    dimension = X[0].shape
    classNumber += 1

    # On ajoute le reste des fichiers numpy de nos classes
    for i, label in enumerate(labels[1:]):
        data = np.load(dataPath + '\\' + label + '.npy')
        X = np.vstack((X, data))
        Y = np.append(Y, np.full(data.shape[0], fill_value=(i+1)))
        classNumber += 1

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, train_size=train_ratio)

    return X_train, X_test, to_categorical(Y_train), to_categorical(Y_test), classNumber, dimension


def get_labels(path):
    labels = [file.replace('.npy', '')
              for file in os.listdir(path) if file.endswith('.npy')]
    label_indices = np.arange(0, len(labels))
    return labels, label_indices, to_categorical(label_indices)


def createDLModel(X_train, X_test, Y_train, Y_test, classNumber, dimension, batch_size, epochs, early, check):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(2, 2), activation='relu',
                     input_shape=(dimension[0], dimension[1], dimension[2])))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(classNumber, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adamax(
                      lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0),
                  metrics=['accuracy'])

    print('Start training model...')
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
              validation_data=(X_test, Y_test), callbacks=[early, check])


def main():
    dataPath = '.\\numpy'
    trainRatio = 0.8
    epochs = 1000
    batch_size = 32
    earlyStopPatience = 5

    early = EarlyStopping(monitor='val_loss', min_delta=0,
                          patience=earlyStopPatience, verbose=0, mode='auto')

    check = ModelCheckpoint('.\\trainedModel\\soundModel.hdf5', monitor='val_loss', verbose=0,
                            save_best_only=True, save_weights_only=False, mode='auto')

    X_train, X_test, Y_train, Y_test, classNumber, dimension = get_train_test(
        trainRatio, dataPath)

    print('DIMENSION X TRAIN ' + str(X_train.shape))
    print('DIMENSION X TEST ' + str(X_test.shape))
    print('DIMENSION Y TRAIN ' + str(Y_train.shape))
    print('DIMENSION Y TEST ' + str(Y_test.shape))

    createDLModel(X_train, X_test, Y_train, Y_test, classNumber, dimension,
                  batch_size, epochs, early, check)


if __name__ == "__main__":
    main()
