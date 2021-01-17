import pyaudio
import wave
import audioop
import time
import os
from tqdm import tqdm
from keras.models import load_model
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
TEST_RECORD_SECONDS = 5
RECORD_TIME = 1
SOUND_PATH = '.\\audiotest.wav'
MODEL_PATH = '.\\trainedModel\\soundModel.hdf5'
DATASET_PATH = '.\\dataset\\'


def test_microphone():

    pyAudioInstance = pyaudio.PyAudio()
    stream = pyAudioInstance.open(format=FORMAT,
                                  channels=CHANNELS,
                                  rate=RATE,
                                  input=True,
                                  frames_per_buffer=CHUNK)

    toneThreshold = 512
    vocalFramesAboveThreshold = 0
    detectedMicrophone = False

    for i in tqdm(range(0, int(RATE / CHUNK * TEST_RECORD_SECONDS)), "> testing microphone..."):
        data = stream.read(CHUNK)
        rms = audioop.rms(data, 2)
        if(rms >= toneThreshold):
            vocalFramesAboveThreshold += 1
            if(vocalFramesAboveThreshold >= 10):
                detectedMicrophone = True
                break

    print("* done testing.")

    if(detectedMicrophone == False):
        print("Microphone not detected.")
    else:
        print("Microphone detected.")

    stream.stop_stream()
    stream.close()
    pyAudioInstance.terminate()

    return detectedMicrophone


def recordAudioSample():
    pyAudioInstance = pyaudio.PyAudio()

    stream = pyAudioInstance.open(format=FORMAT,
                                  channels=CHANNELS,
                                  rate=RATE,
                                  input=True,
                                  frames_per_buffer=CHUNK)

    print("Record will start in 3s!")
    print("3")
    time.sleep(1)
    print("2")
    time.sleep(1)
    print("1...")
    time.sleep(1)
    frames = []
    for i in tqdm(range(0, int(RATE / CHUNK * RECORD_TIME)), "> Recording sample... "):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    pyAudioInstance.terminate()

    print("Record finished. Creating .wav file...")
    wf = wave.open(SOUND_PATH, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pyAudioInstance.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    return


def makePredictionForRecordedAudioSample():
    soundSize = (50, 50)
    labels = []

    for label in os.listdir(DATASET_PATH):
        labels.append(label)

    model = load_model(MODEL_PATH)

    print("\n> Processing audio sample just recorded...")
    start = time.time()

    data = []
    y, sr = librosa.load(SOUND_PATH)

    temp = librosa.feature.melspectrogram(y=y, sr=sr)
    librosa.display.specshow(librosa.power_to_db(temp, ref=np.max))

    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()
    plt.tight_layout(pad=0)
    plt.axis('off')
    img = Image.frombytes(
        'RGB', canvas.get_width_height(), canvas.tostring_rgb())
    img = img.resize(size=soundSize)
    img = np.asarray(img) / 255.
    data.append(img)
    plt.close()
    data = np.asarray(data)

    dimension = data[0].shape
    data = data.astype(np.float32).reshape(
        data.shape[0], dimension[0], dimension[1], dimension[2])

    prediction = model.predict(data)

    # Retrieving label with the highest prediction value
    maxPredict = np.argmax(prediction)

    # Retrieving class name labeled
    print(maxPredict)
    word = labels[maxPredict]
    pred = prediction[0][maxPredict] * 100.
    end = time.time()

    print("Class prediction :")
    for i in range(0, len(labels)):
        print('     ' + labels[i] + ' : ' +
              "{0:.2f}%".format(prediction[0][i] * 100.))

    print('Result : ' + word + ' : ' + "{0:.2f}%".format(pred))
    print('Time spent for prediction : ' + "{0:.2f}secs".format(end - start))
    return


def main():
    # operationalMicrophone = test_microphone()
    # if(operationalMicrophone == False):
    #     print("Can't proceed to record an audio sample if micro is not working.")
    #     return
    print("Be ready to be recorded...")
    recordAudioSample()
    print("Record finished, now lets predict!")
    makePredictionForRecordedAudioSample()


if __name__ == '__main__':
    main()
