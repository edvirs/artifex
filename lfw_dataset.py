import Dataset
import os
import pandas as pd
from Dataset import Dataset

class LWF(Dataset):
  def __init__(self, datasetPath):
    self.datasetPath = datasetPath
    names = pd.read_csv(datasetPath + 'people.csv')
    

  def load_data(self):
    return self.data
  

  def img2data(self):
    pass 
  

  def getImg(self, path):
    pass
