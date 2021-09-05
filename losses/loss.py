import keras.backend as K
import sys
sys.path.append('..')
from lib.config import *
from keras.losses import binary_crossentropy
opt = args()

def rec_M_loss(y_true,y_pred):
    return K.mean(binary_crossentropy(y_true,y_pred))

def rec_f_loss(y_true,y_pred):
    return K.mean(K.abs(y_true-y_pred))

def rec_l_loss(y_true,y_pred):
    return K.mean(binary_crossentropy(y_true,y_pred))

def rec_Sita_loss(x_real,x_rec,fg):
    loss = K.mean(K.abs((x_real-x_rec)*(fg+0.5*(1-fg))))
    return loss

def loss_code_adv_discriminator(y_true,y_pred,label_true,label_fake):
    loss1 = binary_crossentropy(label_true,y_true)
    loss2 = binary_crossentropy(label_fake,y_pred)
    loss = loss1 + loss2
    return K.mean(loss)

def loss_code_adv_generator(label_true,y_true):
    loss = binary_crossentropy(label_true,y_true)
    return K.mean(loss)

def loss_adv_gen_loss(label_true,y_true):
    loss = binary_crossentropy(label_true,y_true)
    return K.mean(loss)

def loss_adv_dis_loss(label_true,y_true,label_fake,y_pred):
    loss1 = binary_crossentropy(label_true, y_true)
    loss2 = binary_crossentropy(label_fake, y_pred)
    loss = loss1 + loss2
    return K.mean(loss)

def loss_patch_adv_gen(label_true,y_true):
    return K.mean(binary_crossentropy(label_true,y_true))

def loss_patch_adv_dis(label_true,y_true,label_fake,y_pred):
    loss1 = K.mean(binary_crossentropy(label_true,y_true))
    loss2 = K.mean(binary_crossentropy(label_fake,y_pred))
    loss = loss1 + loss2
    return loss

def loss_all(y_true,vae_out,generator_out):
    [image, batch_mask, batch_part, batch_landmark, batch_fg] = y_true
    [Cw_z_f, Cw_z_l, Cw_z_M, M_f, X_f, X_l] = vae_out
    [D_fake_patch, D_fake_global, image_out] = generator_out
    label_f  = K.ones_like(Cw_z_f)
    label_M = K.ones_like(Cw_z_M)
    label_l = K.ones_like(Cw_z_l)
    label_p = K.ones_like(D_fake_patch)
    label_g = K.ones_like(D_fake_global)
    rec_m_loss = rec_M_loss(batch_mask,M_f)
    rec_F_loss = rec_f_loss(batch_part,X_f)
    rec_L_loss = rec_l_loss(batch_landmark,X_l)
    rec_sita_loss = rec_Sita_loss(image,image_out,batch_fg)
    lat_f_loss = loss_code_adv_generator(label_f,Cw_z_f)
    lat_l_loss = loss_code_adv_generator(label_l,Cw_z_l)
    lat_m_loss = loss_code_adv_generator(label_M,Cw_z_M)
    adv_patch_g_loss = loss_patch_adv_gen(label_p,D_fake_patch)
    adv_global_g_loss = loss_adv_gen_loss(label_g,D_fake_global)
    all_loss = 4000*(rec_F_loss+rec_sita_loss+rec_m_loss)\
               +2000*rec_L_loss+30*(lat_f_loss+lat_m_loss+lat_l_loss)+30*adv_patch_g_loss+20*adv_global_g_loss
    return all_loss


def loss_part_all(y_true,vae_out,generator_out):
    [image, batch_part, batch_fg] = y_true
    [Cw_z_f, X_f] = vae_out
    [D_fake_patch, D_fake_global, image_out] = generator_out
    label_f  = K.ones_like(Cw_z_f)
    label_p = K.ones_like(D_fake_patch)
    label_g = K.ones_like(D_fake_global)
    rec_F_loss = rec_f_loss(batch_part, X_f)
    rec_sita_loss = rec_Sita_loss(image,image_out,batch_fg)
    lat_f_loss = loss_code_adv_generator(label_f,Cw_z_f)
    adv_patch_g_loss = loss_patch_adv_gen(label_p,D_fake_patch)
    adv_global_g_loss = loss_adv_gen_loss(label_g,D_fake_global)
    all_loss = 4000*(rec_sita_loss+rec_F_loss)+30*lat_f_loss+30*adv_patch_g_loss+20*adv_global_g_loss
    return all_loss


def loss_mask_all(y_true,vae_out,generator_out):
    [image, batch_mask, batch_fg] = y_true
    [Cw_z_f, M_f] = vae_out
    [D_fake_patch, D_fake_global, image_out] = generator_out
    label_f  = K.ones_like(Cw_z_f)
    label_p = K.ones_like(D_fake_patch)
    label_g = K.ones_like(D_fake_global)
    rec_m_loss = rec_M_loss(batch_mask,M_f)
    rec_sita_loss = rec_Sita_loss(image,image_out,batch_fg)
    lat_f_loss = loss_code_adv_generator(label_f,Cw_z_f)
    adv_patch_g_loss = loss_patch_adv_gen(label_p,D_fake_patch)
    adv_global_g_loss = loss_adv_gen_loss(label_g,D_fake_global)
    all_loss = 4000*(rec_sita_loss+rec_m_loss)+30*lat_f_loss+30*adv_patch_g_loss+20*adv_global_g_loss
    return all_loss