import sys
import argparse


import math
import numpy as np
import keras
from matplotlib import pyplot as plt

from keras.layers import Dropout,Conv2D,MaxPooling2D,UpSampling2D,Input
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.optimizers import RMSprop

from sklearn.model_selection import train_test_split



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



# params is a dictionary that specifies the hyperparameters
# it contains the number of convolutional layers                    (numConvLayers)
#             the shape of the convolutional filters                (convFiltersShape)
#             the numbers of the convolutional filters per layer    (numsConvFilters)
#             the number of epochs                                  (epochs)
#             the batch size                                        (batch_size)

def autoencoder(trainData, trainLabels, valData, valLabels, params):

    # extract row and column information
    numberOfRows = trainData.shape[1]
    numberOfColumns = trainData.shape[2]


    # create an empty model
    model = Sequential()
    model.add(Input(shape = (numberOfRows, numberOfColumns, 1)))


    # calculate when to use maxPooling2D
    resizeTimes = int(math.log(numberOfRows / 7, 2))

    # when to pool
    poolingIndices = [2 * (x + 1) - 1 for x in range(resizeTimes)]

    # when to upsample
    upSamplingIndices = [params['numConvLayers'] - 2 * (x + 1) - 1 for x in range(resizeTimes)]


    # for every layer, add encoding layer
    for layerNumber in range(params['numConvLayers']):
        model.add(Conv2D(params['numsConvFilters'][layerNumber], params['convFiltersShape'], activation='relu', padding='same'))
        model.add(BatchNormalization())
        
        #check if we should maxPool
        if layerNumber in poolingIndices and layerNumber != (params['numConvLayers'] - 1):
            model.add(MaxPooling2D(pool_size=(2, 2)))


    # for every layer but one, mirror decoding layer
    for layerNumber in range(params['numConvLayers'] - 1):
        layerDataIndex = -2 - layerNumber
        model.add(Conv2D(params['numsConvFilters'][layerDataIndex], params['convFiltersShape'], activation='relu', padding='same'))
        model.add(BatchNormalization())

        #check if we should upSample
        if layerNumber in upSamplingIndices:
            model.add(UpSampling2D((2,2)))

    # add final layer with sigmoid activation function
    model.add(Conv2D(1, params['convFiltersShape'], activation='sigmoid', padding='same'))




    model.compile(loss = 'mean_squared_error', optimizer = RMSprop())
    
    history = model.fit(trainData, trainLabels, 
                        validation_data=(valData, valLabels),
                        batch_size=params['batch_size'],
                        epochs=params['epochs'],
                        verbose=1)
    
    # finally we have to make sure that history object and model are returned
    return history, model






def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d')
    args = parser.parse_args()

    datasetFilePath = args.d

    # read the data into memory once
    inputData = readImages(datasetFilePath)
    inputData = inputData.reshape(-1, 28,28, 1)
    inputData = inputData / np.max(inputData)

    # initialize parameters
    p = {}

    # initialize plotting data
    plottingData = {}
    for hyperParameterName in ['numConvLayers', 'convFiltersShape', 'numsConvFilters', 'epochs', 'batch_size']:
        plottingData[hyperParameterName] = []


    # start experiment loop
    while True:
        
        # read hyperparameters from user
        print('Please enter the number of convolutional layers.')
        p['numConvLayers'] = int(input())

        print('Please enter the size of the convolutional filter.')
        convSize = int(input())
        p['convFiltersShape'] = (convSize, convSize)


        p['numsConvFilters'] = []
        for layerNumber in range(p['numConvLayers']):
            print('Please enter the number of convolutional filters for layer number ' + str(layerNumber + 1) + '.')
            p['numsConvFilters'].append(int(input()))

        print('Please enter the number of epochs.')
        p['epochs'] = int(input())

        print('Please enter the batch size.')
        p['batch_size'] = int(input())



        train_X,valid_X,train_ground,valid_ground = train_test_split(inputData,
                                                                    inputData,
                                                                    test_size=0.2,
                                                                    random_state=13)


        history, model = autoencoder(train_X, train_ground, valid_X, valid_ground, p)


        # log plotting data for experiment
        loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]

        for hyperParameterName in ['numConvLayers', 'convFiltersShape', 'numsConvFilters', 'epochs', 'batch_size']:
            plottingData[hyperParameterName].append((p[hyperParameterName], loss, val_loss))



        
        print('Do you want to repeat experiment with other hyperparameters? (y/n)')
        repeat = input() == 'y'


        # prompt for plotting
        print('Do you want to plot loss functions against hyperparameters? (y/n)')
        if(input() == 'y'):
            # do the plots
            plt.figure(figsize=(8, 6))
            positionInPlot = 1
            for hyperParameterName in ['numConvLayers', 'convFiltersShape', 'numsConvFilters', 'epochs', 'batch_size']:
                sortedData = sorted(plottingData[hyperParameterName])
                hyperParameter = [str(x) for (x, _, _) in sortedData]
                loss = [y for (_, y, _) in sortedData]
                val_loss = [z for (_, _, z) in sortedData]

                plt.subplot(3, 2, positionInPlot)
                plt.plot(hyperParameter, loss, 'bo', label='Training loss')
                plt.plot(hyperParameter, val_loss, 'b', label='Validation loss')
                plt.title(hyperParameterName)
                plt.legend()
                positionInPlot += 1

            plt.tight_layout()
            plt.show()

        # prompt for saving 
        print('Do you want to save the last model trained? (y/n)')
        if(input() == 'y'):
            print('Please enter the path where the model will be saved.')
            model.save(input())


        if(not repeat):
            break




if __name__ == "__main__":
    main()

