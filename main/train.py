from keras.layers import Input,concatenate
from keras.optimizers import Adam
from losses.loss import loss_all
from model.Unet_model import unet
from keras import Model
import numpy as np
import os
import argparse
# from lib.config import args
from model.vae_model import encoder_F,encoder_L,decoder_F,decoder_L,decoder_M
from model.discriminator_model import discriminator_global,discriminator_patch,Code_f_discriminator,Code_l_discriminator,Code_M_discriminator

from data.data_load import get_train_data,Valid_Data
os.environ['CUDA_VISIBLE_DEVICES']='0'

class DEGAN:
    def __init__(self,args):
        self.encoder_f = encoder_F()
        self.encoder_t = encoder_L()
        self.decoder_M = decoder_M()
        self.decoder_f = decoder_F()
        self.decoder_l = decoder_L()
        self.Unet = unet()
        self.discriminator_global = discriminator_global()
        self.discriminator_patch = discriminator_patch()
        self.discriminator_Cwf = Code_f_discriminator()
        self.discriminator_Cwl = Code_l_discriminator()
        self.discriminator_CMw = Code_M_discriminator()
        self.args = args
        self.model = self.LIF_model()
        self.adam = Adam(lr=args.base_lr, beta_1=0.5, beta_2=0.999)
        self.discriminator_global.compile(loss='binary_crossentropy', optimizer=self.adam)
        self.discriminator_patch.compile(loss='binary_crossentropy', optimizer=self.adam)
        self.discriminator_CMw.compile(loss='binary_crossentropy', optimizer=self.adam)
        self.discriminator_Cwf.compile(loss='binary_crossentropy', optimizer=self.adam)
        self.discriminator_Cwl.compile(loss='binary_crossentropy', optimizer=self.adam)
        self.discriminator_global.trainable = False
        self.discriminator_patch.trainable = False
        self.discriminator_CMw.trainable = False
        self.discriminator_Cwf.trainable = False
        self.discriminator_Cwl.trainable = False

    def DEGAN_Model(self):
        image = Input(shape=(self.args.image_size, self.args.image_size, 3))
        lack_image = Input(shape=(self.args.image_size, self.args.image_size, 3))
        batch_mask = Input(shape=(self.args.image_size, self.args.image_size, 1))
        batch_part = Input(shape=(self.args.image_size, self.args.image_size, 3))
        batch_landmark = Input(shape=(self.args.image_size, self.args.image_size, 1))
        batch_fg = Input(shape=(self.args.image_size, self.args.image_size, 3))
        y_true = [image, batch_mask, batch_part, batch_landmark, batch_fg]
        [Z_f, rand_f] = self.encoder_f(lack_image)
        [Z_l, rand_l] = self.encoder_t(lack_image)
        Z_M = concatenate([Z_f, Z_l], axis=-1)
        rand_M = concatenate([rand_f, rand_l], axis=-1)
        Cw_z_f = self.discriminator_Cwf(Z_f)
        Cw_z_l = self.discriminator_Cwl(Z_l)
        Cw_z_M = self.discriminator_CMw(Z_M)
        M_f = self.decoder_M(Z_M)
        X_f = self.decoder_f(Z_M)
        X_l = self.decoder_l(Z_l)
        vae_out = [Cw_z_f, Cw_z_l, Cw_z_M, M_f, X_f, X_l]
        image_out = self.Unet([lack_image, Z_M])
        fake_out = self.Unet([lack_image, rand_M])
        D_fake_patch = self.discriminator_patch(image_out)
        D_fake_global = self.discriminator_global(image_out)
        geneator_out = [D_fake_patch, D_fake_global, image_out]
        all_loss = loss_all(y_true, vae_out, geneator_out)
        model = Model(inputs=[image, lack_image, batch_mask, batch_part, batch_landmark, batch_fg],
                      outputs=[image_out, fake_out, rand_f, Z_f, rand_l, Z_l, rand_M, Z_M, M_f, X_f, X_l])
        model.add_loss(all_loss)
        model.summary()
        return model

    def train(self):
        self.model.compile(self.adam)
        self.file_names = os.listdir(self.args.image)
        self.valid_file_names = os.listdir(self.args.valid_image)
        self.indexes = np.arange(len(self.file_names))
        self.valid_indexes = np.arange(len(self.valid_file_names))
        real_label = np.ones((self.args.batch_size, 1))
        fake_label = np.zeros((self.args.batch_size, 1))
        patch_real = np.ones((self.args.batch_size, 16, 16, 1))
        patch_fake = np.zeros((self.args.batch_size, 16, 16, 1))
        for epoch in range(self.args.epoch):
            np.random.shuffle(self.indexes)
            np.random.shuffle(self.valid_indexes)
            # Valid_Data(opt, self.valid_file_names, self.valid_indexes, self.model, epoch)
            for item in range(len(self.file_names) // self.args.batch_size):
                [batch_image, batch_lack, batch_mask, batch_part, batch_landmark, batch_fg, irr_mask] = get_train_data(
                    item, self.args, self.indexes, self.file_names)
                [image_out, fake_out, rand_f, Z_f, rand_l, Z_l, rand_M, Z_M, M_f, X_f, X_l] = self.model.predict(
                    [batch_image, batch_lack, batch_mask, batch_part, batch_landmark, batch_fg])
                D_patch_real_loss = self.discriminator_patch.train_on_batch(batch_image, patch_real)
                D_patch_fake_loss = self.discriminator_patch.train_on_batch(image_out, patch_fake)
                D_patch_fake_loss2 = self.discriminator_patch.train_on_batch(fake_out, patch_fake)
                D_patch_loss = D_patch_real_loss + D_patch_fake_loss + D_patch_fake_loss2
                D_global_real_loss = self.discriminator_global.train_on_batch(batch_image, real_label)
                D_global_fake_loss = self.discriminator_global.train_on_batch(image_out, fake_label)
                D_global_fake_loss2 = self.discriminator_global.train_on_batch(fake_out, fake_label)
                D_global_loss = D_global_real_loss + D_global_fake_loss + D_global_fake_loss2
                Cf_real_loss = self.discriminator_Cwf.train_on_batch(rand_f, real_label)
                Cf_fake_loss = self.discriminator_Cwf.train_on_batch(Z_f, fake_label)
                Cf_loss = Cf_real_loss + Cf_fake_loss

                Cl_real_loss = self.discriminator_Cwl.train_on_batch(rand_l, real_label)
                Cl_fake_loss = self.discriminator_Cwl.train_on_batch(Z_l, fake_label)
                Cl_loss = Cl_real_loss + Cl_fake_loss

                CM_real_loss = self.discriminator_CMw.train_on_batch(rand_M, real_label)
                CM_fake_loss = self.discriminator_CMw.train_on_batch(Z_M, fake_label)
                CM_loss = CM_fake_loss + CM_real_loss

                FIL_loss = self.model.train_on_batch(
                    [batch_image, batch_lack, batch_mask, batch_part, batch_landmark, batch_fg], None)
                print(
                    "epoch:%d , iters:%d , D_patch_loss:%f , D_global_loss:%f , C_f_loss:%f , C_l_loss:%f , C_M_loss:%f , FIL_loss:%f,d_real:%f, d_fake:%f, d_rand:%f" % (
                        epoch, item, D_patch_loss, D_global_loss, Cf_loss, Cl_loss, CM_loss, FIL_loss,
                        D_global_real_loss, D_global_fake_loss, D_global_fake_loss2))
            # valid after once epoch
            model_path = os.path.join(self.args.save_model, 'model')
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            self.model.save_weights(model_path + '/' + str(epoch) + 'FIL_weights.h5')
            self.discriminator_CMw.save_weights(model_path + '/' + str(epoch) + 'dis_CMw_weights.h5')
            self.discriminator_Cwf.save_weights(model_path + '/' + str(epoch) + 'dis_CWf_weights.h5')
            self.discriminator_Cwl.save_weights(model_path + '/' + str(epoch) + 'dis_Cwl_weights.h5')
            self.discriminator_global.save_weights(model_path + '/' + str(epoch) + 'dis_global_weights.h5')
            self.discriminator_patch.save_weights(model_path + '/' + str(epoch) + 'dis_patch_weights.h5')
            Valid_Data(opt, self.valid_file_names, self.valid_indexes, self.model, epoch)

if __name__ == '__main__':
    root_dir = 'train/'
    valid_dir = 'valid/'
    test_dir = 'test/'
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default=root_dir + 'img', type=str, help='train image')
    parser.add_argument('--mask', default=root_dir + 'face_mask', type=str, help='mask')
    parser.add_argument('--fg', default=root_dir + 'fg', type=str, help='foreground')
    parser.add_argument('--landmark', default=root_dir + '68_landmark', type=str, help='point of landmark')
    parser.add_argument('--part', default=root_dir + 'face_part', type=str, help='part of face')
    parser.add_argument('--valid_image', default=valid_dir + 'img', help='valid image')
    parser.add_argument('--test_image', type=str, default=test_dir + 'img')
    parser.add_argument('--image_size', default=256, type=int, help='image size')
    parser.add_argument('--epoch', default=200, type=int, help='epochs')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--base_lr', default=2e-4, type=float)
    parser.add_argument('--save_model', default='/home/zhangxian/model/HQ_new/face_inpainting_hq_adv_HQ',
                            help='model')
    opt = parser.parse_args()
    LIF_net = DEGAN(opt)
    LIF_net.train()
