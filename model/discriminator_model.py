from keras.layers import Input,Conv2D,LeakyReLU,Flatten,Dense
from keras import Model

def discriminator_patch():
    inputs = Input(shape=(256,256,3))
    x = Conv2D(32,kernel_size=4,strides=(2,2),padding='same')(inputs)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(64, kernel_size=4, strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(128, kernel_size=4, strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(256, kernel_size=4, strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(1,kernel_size=1,strides=(1,1),activation='sigmoid')(x)
    patch_model = Model(inputs,x)
    return patch_model

def discriminator_global():
    inputs = Input(shape=(256,256,3))
    x = Conv2D(32,kernel_size=4,strides=(2,2),padding='same')(inputs)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(64, kernel_size=4, strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(128, kernel_size=4, strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(256, kernel_size=4, strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(512, kernel_size=4, strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Flatten()(x)
    x = Dense(1,activation='sigmoid')(x)
    global_model = Model(inputs,x)
    return global_model


def Code_f_discriminator():
    input = Input(shape=(256,))
    z = Dense(128,activation='relu')(input)
    z = Dense(128,activation='relu')(z)
    z = Dense(1,activation='sigmoid')(z)
    cw_model = Model(input,z)
    return cw_model

def Code_l_discriminator():
    input = Input(shape=(256,))
    z = Dense(128,activation='relu')(input)
    z = Dense(128,activation='relu')(z)
    z = Dense(1,activation='sigmoid')(z)
    cw_model = Model(input,z)
    return cw_model

def Code_M_discriminator():
    input = Input(shape=(512,))
    z = Dense(128,activation='relu')(input)
    z = Dense(128,activation='relu')(z)
    z = Dense(1,activation='sigmoid')(z)
    cw_model = Model(input,z)
    return cw_model