import math
import pandas as pd
import cv2
from Dataset import Dataset
import matplotlib.pyplot as plt
import numpy as np
import pickle
from alive_progress import alive_bar

class LWF(Dataset):
  indexes = []
  names = []

  def __init__(self, datasetPath):
    self.datasetPath = datasetPath
    self.names = pd.read_csv(datasetPath + 'people.csv')
    self.indexes = self.names["images"].values
    self.names = self.names["name"].values
    self.loadImages(self.names, self.indexes)


  def load_data(self):
    return self.data


  def getImgPath(self, name, index):
    return f"{self.datasetPath}lfw_funneled/{name}/{name}_{index:04d}.jpg"
  

  def loadFromFile(self):
    data = []
  
    with open(self.datasetPath + "dataset", "rb") as file:
      data = pickle.load(file)

    print(np.shape(data))

    return data


  def loadImages(self, names, indexes, save2file = False):
    data = []
    count = len(names)

    print("Loadind images from dataset...")
    with alive_bar(count) as bar:
      for i in range(count):

        if not math.isnan(indexes[i]):

          for j in range(int(indexes[i])):

            path = self.getImgPath(names[i], j + 1)
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255
            data.append(img)

        bar()

    data = np.array(data)

    if save2file:
      print("Saving data into file ...")
      with open(self.datasetPath + "dataset", "wb") as file:
        pickle.dump(data, file)
      print("Done")
    else:
      self.data = data