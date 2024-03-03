import cv2, time, os , tensorflow as tf
import numpy as np

from tensorflow.python.keras.utils.data_utils import get_file

np.random.seed(123)

class Detector:
    def __init__(self):
        pass

    def readClasses(self, classesFilePath):
        with open(classesFilePath, 'r') as f:
            self.classesList = f.read().splitlines()


        self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))    

    def downloadModel(self, modelURL):

        filename = os.path.basename(modelURL)
        self.modelName = filename[:filename.index('.')]

        self.cacheDir = "./pretrained_models"

        os.makedirs(self.cacheDir, exist_ok=True)

        get_file(fname=filename, origin=modelURL, cache_dir=self.cacheDir , cache_subdir="checkpoints" , extract=True)


    def loadModel(self):
        print("Loading Model "+ self.modelName)
        tf.keras.backend.clear_session()
        self.model = tf.saved_model.load(os.path.join(self.cacheDir, "checkpoint" , self.modelName))