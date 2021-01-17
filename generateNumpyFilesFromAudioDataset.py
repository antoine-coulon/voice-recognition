import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import librosa
import librosa.display
import matplotlib.pyplot as plt

numpyPath = '.\\numpy'
datasetPath = '.\\dataset'


def createDataDirsOrSkip():
    if(os.path.exists(numpyPath) == False):
        os.mkdir(numpyPath)
    if(os.path.exists(datasetPath) == False):
        # download .tar.gz from Google repository
        extractDatasetFromURL('http://')

def extractDatasetFromURL(url):
    # TODO
    return {}


def launchConversion(datasetPath, numpyPath, resizeImg, imgSize):
    # Each folder determines a different class.
    # dog/" => will trigger a class of name "dog" and samples will be generated from each audio file contained in it
    for soundClass in os.listdir(datasetPath):
        pathSound = datasetPath + '\\' + soundClass
        imgs = []

        # Pour chaque image d'une classe, on la charge, resize et transforme en tableau
        for soundFile in tqdm(os.listdir(pathSound), "Converting : '{}'".format(soundClass)):
            imgSoundPath = pathSound + '\\' + soundFile

            y, sr = librosa.load(imgSoundPath)

            # Running mel spectogram on each audio file contained in the dataset
            temp = librosa.feature.melspectrogram(y=y, sr=sr)

            librosa.display.specshow(librosa.power_to_db(temp, ref=np.max))
            canvas = plt.get_current_fig_manager().canvas
            canvas.draw()
            img = Image.frombytes(
                'RGB', canvas.get_width_height(), canvas.tostring_rgb())

            if resizeImg == True:
                img = img.resize(size=imgSize)
            plt.tight_layout(pad=0)
            plt.axis('off')
            # spectogram show
            # plt.show()
            data = np.asarray(img, dtype=np.float32)
            imgs.append(data)
            plt.close()

        # Converting pixels from 0 to 255 to gradients from 0 to 1 (numpy array)
        imgs = np.asarray(imgs) / 255.
        np.save(numpyPath + '\\ ' + soundClass + '.npy', imgs)


def main():
    createDataDirsOrSkip()
    resizeImg = True
    # change resolution for better performances
    imgSize = (50, 50)
    launchConversion(datasetPath, numpyPath, resizeImg, imgSize)


if __name__ == '__main__':
    main()
