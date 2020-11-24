import sys
import argparse

import math
import numpy as np
import keras
from matplotlib import pyplot as plt

from keras.layers import Dropout,Conv2D,MaxPooling2D,UpSampling2D,Input,Flatten,Dense
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.optimizers import RMSprop
from keras.utils import to_categorical


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


import pandas as pd


def readImages(filename):
    with open(filename, mode='rb') as bytestream:
        # throw away magic number
        bytestream.read(4)
        # read the number of images from the metadata
        numOfImages = int.from_bytes(bytestream.read(4), byteorder='big')
        # read the number of rows from the metadata
        numOfRows = int.from_bytes(bytestream.read(4), byteorder='big')
        # read the number of columns from the metadata
        numOfColumns = int.from_bytes(bytestream.read(4), byteorder='big')

        # read actual image data
        buf = bytestream.read(numOfRows * numOfColumns * numOfImages)
        # convert data from bytes to numpy array
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        # reshape so we can access the data of an image
        data = data.reshape(numOfImages, numOfRows, numOfColumns)

        return data
    
  
def readLabels(filename):
    with open(filename, mode='rb') as bytestream:
        # throw away magic number
        bytestream.read(4)
        # read the number of labels from the metadata
        numOfLabels = int.from_bytes(bytestream.read(4), byteorder='big')
      
        # read actual label data
        buf = bytestream.read(numOfLabels)
        # convert data from bytes to numpy array
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

        return labels


def buildCompleteModel(trainData, trainLabels, valData, valLabels, params):

  epochs = params['epochs']
  batchSize = params['batch_size']
  FCNeurons = params['FCNeurons']
  savedModelPath = params['savedModelPath']
  outputNeurons = params['outputNeurons']


  # we need to load the saved model and add the encoder layers to a new model
  savedModel = keras.models.load_model(savedModelPath)

  # create a new model and insert the layers
  fullModel = Sequential()

  # loop through all encoder layers which are floor(layers / 2) + 1
  numberOfEncoderLayers = math.floor(len(savedModel.layers) / 2) + 1

  for layerNumber in range(numberOfEncoderLayers):
    fullModel.add(savedModel.layers[layerNumber])
  
  # now we must flatten and connect fully connected layer and output layer
  fullModel.add(Flatten())
  fullModel.add(Dense(FCNeurons, activation='relu'))
  fullModel.add(Dense(outputNeurons, activation='softmax'))



  # now we must train everything but the encoder part
  # disable training in encoder layers
  for layer in fullModel.layers[0:numberOfEncoderLayers]:
    layer.trainable = False

  # compile and train 
  fullModel.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

  classifyTrain = fullModel.fit(trainData, trainLabels, batch_size = batchSize, epochs = math.ceil(epochs * 0.9), verbose = 1, validation_data = (valData, valLabels))


  # and now finally train the whole network

  # now we must train everything
  # enable training in encoder layers
  for layer in fullModel.layers[0:numberOfEncoderLayers]:
    layer.trainable = True

  # compile and train 
  fullModel.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

  classifyTrain = fullModel.fit(trainData, trainLabels, batch_size = batchSize, epochs = math.ceil(epochs * 0.1), verbose = 1, validation_data = (valData, valLabels))


  

  return classifyTrain, fullModel



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d')
    parser.add_argument('-dl')
    parser.add_argument('-t')
    parser.add_argument('-tl')
    parser.add_argument('-model')
    args = parser.parse_args()

    datasetFilePath = args.d
    datasetLabelsFilePath = args.dl

    testsetFilePath = args.t
    tessetLabelsFilePath = args.tl

    modelFilePath = args.model

    # read the data into memory once
    inputData = readImages(datasetFilePath)
    inputData = inputData.reshape(-1, 28,28, 1)
    inputData = inputData / np.max(inputData)

    testData = readImages(testsetFilePath)
    testData = testData.reshape(-1, 28,28, 1)
    testData = testData / np.max(testData)

    # read the labels into memory
    inputLabels = readLabels(datasetLabelsFilePath)
    testLabels = readLabels(tessetLabelsFilePath)

    # split with sklearn
    trainData, valData, trainLabels, valLabels = train_test_split(inputData,
                                                                  inputLabels,
                                                                  test_size=0.2,
                                                                  random_state=13)


    # initialize parameters
    p = {}

    # add trivial values
    p['outputNeurons'] = int(max(inputLabels)) + 1
    p['savedModelPath'] = modelFilePath


    # initialize plotting data
    plottingData = {}
    for hyperParameterName in ['FCNeurons', 'epochs', 'batch_size']:
        plottingData[hyperParameterName] = []

    # start experiment loop
    while True:
        
        # read hyperparameters from user
        print('Please enter the number of neurons for fully connected layer.')
        p['FCNeurons'] = int(input())

        print('Please enter the number of epochs.')
        p['epochs'] = int(input())

        print('Please enter the batch size.')
        p['batch_size'] = int(input())

        # build and train the model
        history, fullModel = buildCompleteModel(trainData, to_categorical(trainLabels), valData, to_categorical(valLabels), p)


        # log plotting data for experiment
        loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]
        acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]

        predictions = [np.argmax(oneHot) for oneHot in fullModel.predict(testData)]

        for hyperParameterName in ['FCNeurons', 'epochs', 'batch_size']:
            metrics = {}
            metrics['loss'] = loss
            metrics['val_loss'] = val_loss
            metrics['acc'] = acc
            metrics['val_acc'] = val_acc
            metrics['precision'] = precision_score(testLabels, predictions, average = 'macro')
            metrics['recall'] = recall_score(testLabels, predictions, average = 'macro')
            metrics['f1'] = f1_score(testLabels, predictions, average = 'macro')
            plottingData[hyperParameterName].append((p[hyperParameterName], metrics))



        
        print('Do you want to repeat experiment with other hyperparameters? (y/n)')
        repeat = input() == 'y'

        # prompt for plotting
        print('Do you want to plot loss functions against hyperparameters? (y/n)')
        if(input() == 'y'):

            # initialise the tables
            tables = {}
            for hyperParameterName in ['FCNeurons', 'epochs', 'batch_size']:
                tables[hyperParameterName] = []

            # do the plots
            plt.figure(figsize=(8, 6))
            positionInPlot = 1
            for hyperParameterName in ['FCNeurons', 'epochs', 'batch_size']:
                plottingData[hyperParameterName].sort(key = lambda tup : tup[0])

                sortedData = plottingData[hyperParameterName]
                
                hyperParameter = [str(x) for (x, _) in sortedData]
                loss = [metrics['loss'] for (_, metrics) in sortedData]
                val_loss = [metrics['val_loss'] for (_, metrics) in sortedData]
                acc = [metrics['acc'] for (_, metrics) in sortedData]
                val_acc = [metrics['val_acc'] for (_, metrics) in sortedData]


                precisionArray = [metrics['precision'] for (_, metrics) in sortedData]
                recallArray = [metrics['recall'] for (_, metrics) in sortedData]
                f1Array = [metrics['f1'] for (_, metrics) in sortedData]

                tables[hyperParameterName].append(hyperParameter)
                tables[hyperParameterName].append(precisionArray)
                tables[hyperParameterName].append(recallArray)
                tables[hyperParameterName].append(f1Array)

                plt.subplot(3, 1, positionInPlot)
                plt.plot(hyperParameter, loss, 'bo', label='Training loss')
                plt.plot(hyperParameter, val_loss, 'b', label='Validation loss')
                plt.plot(hyperParameter, acc, 'ro', label='Accuracy')
                plt.plot(hyperParameter, val_acc, 'r', label='Validation accuracy')
                plt.title(hyperParameterName)
                plt.legend()
                positionInPlot += 1

            plt.tight_layout()
            plt.show()


            for hyperParameterName in ['FCNeurons', 'epochs', 'batch_size']:
                print('Table for Hyperparameter: ' + hyperParameterName)
                table = pd.DataFrame(tables[hyperParameterName])
                table.index = ['Value', 'Precision', 'Recall', 'F1']
                print(table)
                print('\n')

        # prompt for saving 
        print('Do you want to classify the test set? (y/n)')
        if(input() == 'y'):
           print('Do you want to use different hyperparameters than last time? (y/n)')
           if(input() == 'y'):
                # read hyperparameters from user
                print('Please enter the number of neurons for fully connected layer.')
                p['FCNeurons'] = int(input())

                print('Please enter the number of epochs.')
                p['epochs'] = int(input())

                print('Please enter the batch size.')
                p['batch_size'] = int(input())

                # build and train the model
                history, fullModel = buildCompleteModel(trainData, to_categorical(trainLabels), valData, to_categorical(valLabels), p)
           classify(fullModel, testData)
            

        if(not repeat):
            break


def classify(model, testData):
    predictions = [np.argmax(oneHot) for oneHot in model.predict(testData)]

    numberOfImages = len(predictions)

    scale = math.ceil(math.sqrt(numberOfImages))
    
    plt.figure(figsize = (16, 12))
    for imageNumber in range(numberOfImages):
        fig = plt.subplot(scale, scale, imageNumber + 1)
        plt.imshow(testData[imageNumber].reshape(28,28), cmap='gray', interpolation='none')
        plt.ylabel("{}".format(predictions[imageNumber]), rotation = 0)
        fig.axes.get_xaxis().set_visible(False)
        fig.yaxis.set_ticks([])
    

    plt.show()
    

if __name__ == "__main__":
    main()
