from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import UpSampling2D, Cropping2D
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.initializers import RandomNormal, VarianceScaling
import numpy as np

# Inspired by https://github.com/pietz/unet-keras (adds padding) and https://github.com/mirzaevinom/promise12_segmentation (adds variance scaling)
# For animations on convulotions : https://github.com/vdumoulin/conv_arithmetic is a great source

"""
	Useful definitions : 
	- Model groups layers into an object with training and inference features
	- Input() is used to instantiate a Keras tensor
	- concatenate() Functional interface to the Concatenate layer
	- Concatenate layer : Layer that concatenates a list of inputs
	- Conv2D : 2D convolutional layer
	- inc_rate: rate at which the conv channels will increase
	- MaxPooling2D : Global max pooling operation for spatial data
	- Conv2DTranspose : Transposed convolution layer (sometimes called Deconvolution)
	- USampling2D : Repeats the rows and columns of the data by size[0] and size[1] respectively
	- Cropping2D : Cropping layer for 2D input
	- RandomNormal : initializer that generates tensors with a normal distribution 
	- VarianceScaling : Initializer capable of adapting its scale to the shape of weights tensors
	- residual: add residual connections around each conv block if true

"""


"""

	[(W−K+2P)/S]+1.
	W is the input volume - in your case 128
	K is the Kernel size - in your case 5
	P is the padding - in your case 0 i believe
	S is the stride - which you have not provided.

"""

def conv_block(m, dim, acti, bn, res, do=0):

    init = VarianceScaling(scale=1.0/9.0) 
    n = Conv2D(dim, 3, activation=acti, padding='same', kernel_initializer=init)(m)
    n = BatchNormalization()(n) if bn else n
    n = Dropout(do)(n) if do else n
    n = Conv2D(dim, 3, activation=acti, padding='same', kernel_initializer=init)(n)
    n = BatchNormalization()(n) if bn else n

    return concatenate([n, m], axis=3) if res else n

def level_block(m, dim, depth, inc, acti, do, bn, mp, up, res):
	if depth > 0:
		n = conv_block(m, dim, acti, bn, res)
		m = MaxPooling2D()(n) if mp else Conv2D(dim, 3, strides=2, padding='same')(n)
		m = level_block(m, int(inc*dim), depth-1, inc, acti, do, bn, mp, up, res)
		if up:
			m = UpSampling2D()(m)
			m = Conv2D(dim, 2, activation=acti, padding='same')(m)
		else:
			m = Conv2DTranspose(dim, 3, strides=2, activation=acti, padding='same')(m)
		n = concatenate([n, m], axis=3)
		m = conv_block(n, dim, acti, bn, res)
	else:
		m = conv_block(m, dim, acti, bn, res, do)
	return m


def UNet(img_shape, out_ch=1, start_ch=64, depth=4, inc_rate=2., activation='relu',
		 dropout=0.0, batchnorm=False, maxpool=True, upconv=False, residual=False):

	i = Input(shape=img_shape)
	o = level_block(i, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual)
	o = Conv2D(out_ch, 1, activation='sigmoid')(o)
	return Model(inputs=i, outputs=o)


