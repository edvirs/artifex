from keras import Model
from keras import saving
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Input
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from lfw_dataset import LWF

class Artifex():
  model_input_shape = (240, 240, 3)

  def __init__(self, encoder = None, decoder= None):
    num_filters = 32

    # if(encoder == None or decoder == None):
    self.encoder = self.getEncoder(self.model_input_shape, numFilters = num_filters)
    #encoder_output_shape = encoder.get_layer("conv2d_9")
    self.decoder = self.getDecoder(numFilters = num_filters)
    # else:
    # self.encoder = encoder
    # self.decoder = decoder
         
    x = self.encoder.output
    output = self.decoder(x)


    self.model = Model(self.encoder.input , output, name = "Artifex")

    self.model.summary()
    self.model.compile(optimizer = "Adam", loss = "mse", metrics=['accuracy'])
    

  def getEncoder(self, input_shape, numFilters = 32):
    input = Input(input_shape)

    x = Conv2D(numFilters, 3, activation='relu', padding='same', kernel_initializer='HeNormal')(input)
    x = Conv2D(numFilters, 3, activation='relu', padding='same', kernel_initializer='HeNormal')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(numFilters*2, 3, activation='relu', padding='same', kernel_initializer='HeNormal')(x)
    x = Conv2D(numFilters*2, 3, activation='relu', padding='same', kernel_initializer='HeNormal')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(numFilters*4, 3, activation='relu', padding='same', kernel_initializer='HeNormal')(x)
    x = Conv2D(numFilters*4, 3, activation='relu', padding='same', kernel_initializer='HeNormal')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(numFilters*8, 3, activation='relu', padding='same', kernel_initializer='HeNormal')(x)
    x = Conv2D(numFilters*8, 3, activation='relu', padding='same', kernel_initializer='HeNormal')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(numFilters*16, 3, activation='relu', padding='same', kernel_initializer='HeNormal')(x)
    output = Conv2D(numFilters*16, 3, activation='relu', padding='same', kernel_initializer='HeNormal')(x)

    model = Model(input, output, name = "encoder")
    model.compile(optimizer = "Adam", loss = "mse", metrics=['accuracy'])

    return model


  def diffusion():
    pass 


  def getDecoder(self, input_shape = (15, 15, 512) , numFilters = 32):
    input = Input(input_shape)
    x = Conv2DTranspose(numFilters*8, (3,3), strides=(2,2), padding='same')(input)
    x = Conv2D(numFilters*8, 3, activation='relu', padding='same', kernel_initializer='HeNormal')(x)
    x = Conv2D(numFilters*8, 3, activation='relu', padding='same', kernel_initializer='HeNormal')(x)

    x = Conv2DTranspose(numFilters*4, (3,3), strides=(2,2), padding='same')(x)
    x = Conv2D(numFilters*4, 3, activation='relu', padding='same', kernel_initializer='HeNormal')(x)
    x = Conv2D(numFilters*4, 3, activation='relu', padding='same', kernel_initializer='HeNormal')(x)

    x = Conv2DTranspose(numFilters*2, (3,3), strides=(2,2), padding='same')(x)
    x = Conv2D(numFilters*2, 3, activation='relu', padding='same', kernel_initializer='HeNormal')(x)
    x = Conv2D(numFilters*2, 3, activation='relu', padding='same', kernel_initializer='HeNormal')(x)

    x = Conv2DTranspose(numFilters, (3,3), strides=(2,2), padding='same')(x)
    x = Conv2D(numFilters, 3, activation='relu', padding='same', kernel_initializer='HeNormal')(x)
    x = Conv2D(numFilters, 3, activation='relu', padding='same', kernel_initializer='HeNormal')(x)
    x = Conv2D(numFilters, 3, activation='relu', padding='same', kernel_initializer='HeNormal')(x)
    output = Conv2D(3, 1, padding='same')(x)

    model = Model(input, output, name = "decoder")
    model.compile(optimizer = "Adam", loss = "mse", metrics=['accuracy'])

    return model


def main():
  print(tf.sysconfig.get_build_info())
  devices = tf.config.list_physical_devices("GPU")
  print(devices)

  dataset = LWF("dataset/", count = 600)
  data = dataset.load_data()
  print(data.shape)

  encoder = saving.load_model("models/encoder.keras")
  decoder = saving.load_model("models/decoder.keras")

  artifex = Artifex(encoder, decoder)
  #artifex = Artifex()
  artifex.model.fit(data, data, epochs=10, batch_size = 32)

  artifex.model.save("models/unet.keras")
  artifex.encoder.save("models/encoder.keras")
  artifex.decoder.save("models/decoder.keras")

def test():
  print(tf.sysconfig.get_build_info())
  devices = tf.config.list_physical_devices("GPU")
  print(devices)

  decoder = saving.load_model("models/decoder.keras")

  img = decoder.predict(np.random.randn(1, 15, 15, 512)).reshape((240, 240, 3))
  print(img.shape)
  plt.imshow(img)

  
if __name__ == "__main__":
  #main()
  test()