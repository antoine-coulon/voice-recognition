import os

trainedModelPath = '../trainedModel/soundModel.hdf5'
    
def removeTrainedModel():
    try:
        os.remove(trainedModelPath)
    except OSError as e:
        print("Error: %s : %s" % (trainedModelPath, e.strerror))


if __name__ == '__main__':
    removeTrainedModel()