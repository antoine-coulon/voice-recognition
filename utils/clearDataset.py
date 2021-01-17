import os
import glob

files = glob.glob('../dataset/**/*.wav', recursive=True)

def clearDataset():
    for f in files:
        try:
            os.remove(f)
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))


if __name__ == '__main__':
    clearDataset()