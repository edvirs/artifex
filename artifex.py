import tensorflow as tf 
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras import layers
import numpy as np

from lfw_dataset import LWF

def main():
  dataset = LWF("dataset/")
  #dataset.loadImages2file(dataset.names, dataset.indexes)
  dataset.loadFromFile()


if __name__ == "__main__":
  main()