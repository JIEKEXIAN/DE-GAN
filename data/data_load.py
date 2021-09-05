import sys
sys.path.append('..')
from keras.utils import Sequence
import os
import numpy as np
from skimage.io import imsave
from skimage.io import imread
from lib.config import *
from random import randint
import cv2


def Valid_Data(opt,file_names,indexs,model,epoch):
    valid_images = np.array([imread(os.path.join(opt.valid_image,file_names[indexs[i]])) for i in range(100)])
    valid_images = valid_images/255
    valid_lack = valid_images.copy()
    real_imgs = valid_images.copy()
    irr_mask = mask_sample(100, opt.image_size, opt.image_size)
    valid_lack = valid_lack * irr_mask
    vaild = valid_lack[:, :, :, 0]
    vaild = np.expand_dims(vaild, axis=-1)
    [image_out, fake_out, rand_f, Z_f, rand_l, Z_l, rand_M, Z_M, M_f, X_f,X_l]=model.predict([valid_lack,valid_lack,vaild,valid_lack,vaild,valid_lack])
    # valid_images[:,64:64+128,64:64+128,:]= image_out[:,64:64+128,64:64+128,:]
    complete_img = valid_lack+(1-irr_mask)*image_out
    # real_path = os.path.join(opt.save_model,str(epoch),'real')
    img_path = os.path.join(opt.save_model,str(epoch),'img')
    mask_path = os.path.join(opt.save_model,str(epoch),'mask')
    face_path = os.path.join(opt.save_model,str(epoch),'face_part')
    land_path = os.path.join(opt.save_model,str(epoch),'land_mark')
    lack_path = os.path.join(opt.save_model,str(epoch),'lack')
    if not os.path.exists(img_path):
        # os.makedirs(real_path)
        os.makedirs(img_path)
        os.makedirs(lack_path)
        os.makedirs(mask_path)
        os.makedirs(face_path)
        os.makedirs(land_path)
    for i in range(100):
        # real_img = real_imgs[i]
        img = complete_img[i]
        lack = valid_lack[i]
        mask = M_f[i,:,:,0]
        face_part = X_f[i]
        land_mark = X_l[i,:,:,0]
        # imsave(os.path.join(real_path,file_names[indexs[i]]),real_img)
        imsave(os.path.join(lack_path, file_names[indexs[i]]), valid_lack[i])
        imsave(os.path.join(img_path, file_names[indexs[i]]), img)
        imsave(os.path.join(mask_path, file_names[indexs[i]]), mask)
        imsave(os.path.join(face_path, file_names[indexs[i]]), face_part)
        imsave(os.path.join(land_path, file_names[indexs[i]]), land_mark)

def mask_sample(batch_size,height,width):
    irr_mask = np.zeros(shape=(batch_size,width,height,3))
    for i in range(batch_size):
        img = np.zeros((width,height))
        mask = np.zeros((height//2,width//2))
        size = int((width//2+height//2)*0.3)
        for _ in range(randint(1,20)):
            x1,x2 = randint(1,width//2),randint(1,width//2)
            y1,y2 = randint(1,height//2),randint(1,height//2)
            thickness = randint(3,size)
            cv2.line(mask,(x1,y1),(x2,y2),(1,1,1),thickness)

        for _ in range(randint(1,20)):
            x1,y1 = randint(1,width//2),randint(1,height//2)
            radius = randint(3,size)
            cv2.circle(mask,(x1,y1),radius,(1,1,1),(-1))

        for _ in range(randint(1,20)):
            x1,y1 = randint(1,width//2),randint(1,height//2)
            s1,s2 = randint(1,width//2),randint(1,height//2)
            a1,a2,a3 = randint(3,180),randint(3,180),randint(3,180)
            thickness = randint(3,size)
            cv2.ellipse(mask,(x1,y1),(s1,s2),a1,a2,a3,(1,1,1),thickness)
        img[width//4:width//4+width//2,height//4:height//4+height//2] = mask
        img = 1 - img
        img = np.expand_dims(img,axis=-1)
        img = np.tile(img,(1,1,3))
        irr_mask[i]=img
    return irr_mask

def get_train_data(item,opt,indexes,file_names):
    batch_indexs = indexes[item *opt.batch_size:(item + 1) *opt.batch_size]
    batch_image = np.array([imread(os.path.join(opt.image, file_names[index])) for index in batch_indexs])
    batch_image = batch_image / 255
    batch_lack = batch_image.copy()
    irr_mask = mask_sample(opt.batch_size,opt.image_size,opt.image_size)
    batch_lack = batch_lack*irr_mask
    batch_mask = np.array([imread(os.path.join(opt.mask, file_names[index])) for index in batch_indexs])
    batch_mask = batch_mask / 255
    batch_mask = np.expand_dims(batch_mask,axis=-1)
    batch_fg = np.array([imread(os.path.join(opt.fg, file_names[index])) for index in batch_indexs])
    batch_fg = batch_fg / 255
    batch_landmark = np.array([imread(os.path.join(opt.landmark, file_names[index])) for index in batch_indexs])
    batch_landmark = batch_landmark / 255
    batch_landmark = np.expand_dims(batch_landmark,axis=-1)
    batch_part = np.array([imread(os.path.join(opt.part,file_names[index])) for index in batch_indexs])
    batch_part = batch_part / 255
    return [batch_image, batch_lack, batch_mask, batch_part, batch_landmark, batch_fg,irr_mask]