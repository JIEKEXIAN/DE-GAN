import keras
from keras.layers import Input,Dense,Lambda,Conv2D,Flatten,LeakyReLU,BatchNormalization,Deconv2D,multiply,add
from keras.models import Model
from keras import backend as k
from keras.layers import Layer

class ScaleShift(Layer):
    def __init__(self,**kwargs):
        super(ScaleShift, self).__init__(**kwargs)
    def call(self, inputs):
        mean,var,rand = inputs
        z = var*rand+mean
        return z

def encoder_F():
    inputs=Input(shape=(256,256,3))
    x= Conv2D(32,kernel_size=4,strides=(2,2),padding='same')(inputs)
    x= LeakyReLU(0.2)(x)
    x = Conv2D(64, kernel_size=4, strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(128, kernel_size=4, strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(256, kernel_size=4, strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(512, kernel_size=4, strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Flatten()(x)
    z_mean = Dense(256,activation='softplus')(x)
    z_std = Dense(256,activation='softplus')(x)
    samples = Lambda(lambda z: k.random_normal(shape=k.shape(z)))(z_mean)
    Z = ScaleShift()([z_mean,z_std,samples])
    FEf_model=Model(inputs,[Z,samples])
    return FEf_model

def encoder_L():
    inputs=Input(shape=(256,256,3))
    x= Conv2D(32,kernel_size=4,strides=(2,2),padding='same')(inputs)
    x= LeakyReLU(0.1)(x)
    x = Conv2D(64, kernel_size=4, strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(128, kernel_size=4, strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(256, kernel_size=4, strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(512, kernel_size=4, strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Flatten()(x)
    z_mean = Dense(256,activation='softplus')(x)
    z_std = Dense(256,activation='softplus')(x)
    samples = Lambda(lambda z: k.random_normal(shape=k.shape(z)))(z_mean)
    Z = ScaleShift()([z_mean,z_std,samples])
    FEf_model=Model(inputs,[Z,samples])
    return FEf_model

def encoder_M():
    inputs=Input(shape=(256,256,3))
    x= Conv2D(32,kernel_size=4,strides=(2,2),padding='same')(inputs)
    x= LeakyReLU(0.1)(x)
    x = Conv2D(64, kernel_size=4, strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(128, kernel_size=4, strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(256, kernel_size=4, strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(512, kernel_size=4, strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Flatten()(x)
    z_mean = Dense(256,activation='softplus')(x)
    z_std = Dense(256,activation='softplus')(x)
    samples = Lambda(lambda z: k.random_normal(shape=k.shape(z)))(z_mean)
    Z = ScaleShift()([z_mean,z_std,samples])
    FEf_model=Model(inputs,[Z,samples])
    return FEf_model

def decoder_M():
    inputs = Input(shape=(512,))
    x = keras.layers.Reshape(target_shape=(16, 16, 2))(inputs)
    x = Deconv2D(128, kernel_size=4, strides=(2, 2), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Deconv2D(64, kernel_size=4, strides=(2, 2), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Deconv2D(32, kernel_size=4, strides=(2, 2), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Deconv2D(16, kernel_size=4, strides=(2, 2), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(1, kernel_size=1, strides=(1, 1), padding='same', activation='sigmoid')(x)
    decoder_model = Model(inputs, x)
    return decoder_model

def decoder_F():
    # inputs = Input(shape=(512,))
    inputs = Input(shape=(256,))
    # x = keras.layers.Reshape(target_shape=(16,16, 2))(inputs)
    x = keras.layers.Reshape(target_shape=(16, 16, 1))(inputs)
    x = Deconv2D(128, kernel_size=4, strides=(2, 2), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Deconv2D(64, kernel_size=4, strides=(2, 2), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Deconv2D(32, kernel_size=4, strides=(2, 2), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Deconv2D(16, kernel_size=4, strides=(2, 2), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(3, kernel_size=1, strides=(1, 1), padding='same', activation='sigmoid')(x)
    decoder_model = Model(inputs, x)
    return decoder_model

def decoder_L():
    inputs = Input(shape=(256,))
    x = keras.layers.Reshape(target_shape=(16,16, 1))(inputs)
    x = Deconv2D(128, kernel_size=4, strides=(2, 2), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Deconv2D(64, kernel_size=4, strides=(2, 2), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Deconv2D(32, kernel_size=4, strides=(2, 2), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Deconv2D(16, kernel_size=4, strides=(2, 2), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(1, kernel_size=1, strides=(1, 1), padding='same', activation='sigmoid')(x)
    decoder_model = Model(inputs, x)
    return decoder_model
