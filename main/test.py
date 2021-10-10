import keras
from main.train import DEGAN
import argparse
from keras import Model
from keras.layers import Input,concatenate
from model.vae_model import encoder_F,encoder_L,decoder_F,decoder_L,decoder_M
from model.Unet_model import unet
import os
from skimage.io import imread
from matplotlib.image import imsave
image_size=256
import numpy as np
from skimage.metrics import peak_signal_noise_ratio,structural_similarity
import matplotlib.pyplot as plt
def DE_GAN():
    encoder_f = encoder_F()
    encoder_t = encoder_L()
    decoder_m = decoder_M()
    decoder_f = decoder_F()
    decoder_l = decoder_L()
    Unet = unet()
    image = Input(shape=(image_size, image_size, 3))
    lack_image = Input(shape=(image_size,image_size, 3))
    batch_mask = Input(shape=(image_size, image_size, 1))
    batch_part = Input(shape=(image_size, image_size, 3))
    batch_landmark = Input(shape=(image_size, image_size, 1))
    batch_fg = Input(shape=(image_size, image_size, 3))
    [Z_f, rand_f] = encoder_f(lack_image)
    [Z_l, rand_l] = encoder_t(lack_image)
    Z_M = concatenate([Z_f, Z_l], axis=-1)
    rand_M = concatenate([rand_f, rand_l], axis=-1)
    M_f = decoder_m(Z_M)
    X_f = decoder_f(Z_M)
    X_l = decoder_l(Z_l)
    image_out = Unet([lack_image, Z_M])
    fake_out = Unet([lack_image, rand_M])
    model = Model(inputs=[image, lack_image, batch_mask, batch_part, batch_landmark, batch_fg],
                  outputs=[image_out, fake_out, rand_f, Z_f, rand_l, Z_l, rand_M, Z_M, M_f, X_f, X_l])
    model.summary()
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('p',"--path",type=str)
    parser.add_argument('-s','--save_path',type=str)
    args = parser.parse_args()
    model = DE_GAN()
    model.load_weights("../checkpoints/DE_GAN.h5")
    imgdir = os.listdir(args.path)
    psnr_list = []
    ssim_list = []
    for imgName in imgdir:
        imgpath = os.path.join(args.path,imgName)
        image = imread(imgpath)
        image = image/255
        image = np.expand_dims(image,axis=0)
        crop = image.copy()
        crop[:,64:64+128,64:64+128,:]=0.0
        mask = np.zeros(shape=(1,image_size,image_size,1))
        mask[:,64:64+128,64:64+128,:]=1.0
        out,_,_,_,_,_,_,_,_,_,_ = model.predict([crop,crop,mask,crop,mask,crop],batch_size=1)
        mask = np.squeeze(mask)
        mask = np.expand_dims(mask,axis=-1)
        image = np.squeeze(image)
        out = np.squeeze(out)
        result = image*(1-mask)+out*mask
        psnr_list.append(peak_signal_noise_ratio(image,result))
        ssim_list.append(structural_similarity(image,result,multichannel=True))
        imsave(os.path.join(args.save_path,imgName),result)

    print("PSNR:",np.mean(psnr_list))
    print("ssim:",np.mean(ssim_list))
