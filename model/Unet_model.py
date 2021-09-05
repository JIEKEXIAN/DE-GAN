from keras.layers import Conv2DTranspose,Conv2D,Input,concatenate,Reshape
from keras import Model

def unet():
    inputs = Input(shape=(256, 256, 3))
    # z = Input(shape=(512,))
    # z_in = Reshape(target_shape=(16, 16, 2))(z)
    z = Input(shape=(256,))
    z_in = Reshape(target_shape=(16, 16,1))(z)
    conv1 = Conv2D(32, kernel_size=3, strides=(1, 1), padding='same', activation='relu')(inputs)
    conv1 = Conv2D(32, kernel_size=3, strides=(1, 1), padding='same', activation='relu')(conv1)

    conv2 = Conv2D(64, kernel_size=3, strides=(2, 2), padding='same', activation='relu')(conv1)
    conv2 = Conv2D(64, kernel_size=3, strides=(1, 1), padding='same', activation='relu')(conv2)

    conv3 = Conv2D(128, kernel_size=3, strides=(2, 2), padding='same', activation='relu')(conv2)
    conv3 = Conv2D(128, kernel_size=3, strides=(1, 1), padding='same', activation='relu')(conv3)

    conv4 = Conv2D(256, kernel_size=3, strides=(2, 2), padding='same', activation='relu')(conv3)
    conv4 = Conv2D(256, kernel_size=3, strides=(1, 1), padding='same', activation='relu')(conv4)

    conv5 = Conv2D(512, kernel_size=3, strides=(2, 2), padding='same', activation='relu')(conv4)
    conv5 = Conv2D(512, kernel_size=3, strides=(1, 1), padding='same', activation='relu')(conv5)

    concat0 = concatenate([conv5, z_in], axis=-1)

    deconv_4 = Conv2DTranspose(256, kernel_size=3, strides=(2, 2), padding='same', activation='relu')(concat0)
    concat1 = concatenate([conv4, deconv_4], axis=-1)

    conv7_1 = Conv2D(256, kernel_size=3, strides=(1, 1), padding='same', activation='relu')(concat1)


    deconv_3 = Conv2DTranspose(128, kernel_size=3, strides=(2, 2), padding='same', activation='relu')(conv7_1)
    concat2 = concatenate([conv3, deconv_3], axis=-1)
    conv8_1 = Conv2D(128, kernel_size=3, strides=(1, 1), padding='same', activation='relu')(concat2)

    deconv_2 = Conv2DTranspose(64, kernel_size=3, strides=(2, 2), padding='same', activation='relu')(conv8_1)
    concat3 = concatenate([conv2,deconv_2],axis=-1) #128

    conv9_1 = Conv2D(64, kernel_size=3, strides=(1, 1), padding='same', activation='relu')(concat3)

    deconv_1 = Conv2DTranspose(32, kernel_size=3, strides=(2, 2), padding='same', activation='relu')(conv9_1)
    concat4 = concatenate([conv1,deconv_1],axis=-1)

    conv10_1 = Conv2D(32, kernel_size=3, strides=(1, 1), padding='same', activation='relu')(concat4)
    result = Conv2D(3, kernel_size=3, strides=(1, 1), padding='same', activation='sigmoid')(conv10_1)
    unet_model = Model([inputs,z],result)
    return unet_model