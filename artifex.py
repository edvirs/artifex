from keras import Model
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Input

from lfw_dataset import LWF

class Artifex():
  model_input_shape = (240, 240, 3)

  def __init__(self):

    num_filters = 32
    encoder = self.getEncoder(self.model_input_shape, numFilters = num_filters)
    #encoder_output_shape = encoder.get_layer("conv2d_9")
    decoder = self.getDecoder(numFilters = num_filters)
         
    x = encoder.output
    x = decoder(x)
    x = Conv2D(num_filters, 3, activation='relu', padding='same', kernel_initializer='HeNormal')(x)
    output = Conv2D(3, 1, padding='same')(x)

    self.model = Model(encoder.input , output, name = "Artifex")

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
    output = Conv2D(numFilters, 3, activation='relu', padding='same', kernel_initializer='HeNormal')(x)

    model = Model(input, output, name = "decoder")

    return model


def main():
  dataset = LWF("dataset/")
  data = dataset.load_data()
  print(data.shape)

  artifex = Artifex()
  artifex.model.fit(data, data, epochs=1)

  
if __name__ == "__main__":
  main()