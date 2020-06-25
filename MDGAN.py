from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Convolution2D
from keras.models import Sequential, Model, load_model
from keras import backend as K
from keras.layers.merge import _Merge
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import sys
import numpy as np
from functools import partial

import numpy as np
from keras.optimizers import Adam, RMSprop
from keras import backend
import keras.backend as K
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.layers.convolutional import  Conv1D, UpSampling1D, MaxPooling1D, Cropping1D
from keras.layers.convolutional import Conv2D, UpSampling2D, MaxPooling2D, Cropping2D, Conv1D,  MaxPooling1D
from keras.layers import Activation, Dense, BatchNormalization, Dropout, ZeroPadding2D
from keras.layers import Input, Reshape, Flatten, Concatenate, Lambda, RepeatVector, Permute
from keras.constraints import Constraint
import matplotlib.pyplot as plt

BATCH_SIZE = 64
GRADIENT_PENALTY_WEIGHT = 10
TRAINING_RATIO = 5
class GAN_GP():
    
    def __init__(self):
        #self.path = "/volumes/data/dataset/gan/MNIST/wgan-gp/wgan-gp_generated_images/"
        #self.path = drive_root_dir+"images/"
        #mnistデータ用の入力データサイズ
        self.img_rows = 28 
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.output=64
        # 潜在変数の次元数 
        self.z_dim = 50
        self.n_critic = 5
        
        # 画像保存の際の列、行数
        self.row = 5
        self.col = 5
        self.row2 = 1 # 連続潜在変数用
        self.col2 = 10# 連続潜在変数用 

        
        # 画像生成用の固定された入力潜在変数
        self.noise_fix1 = np.random.uniform(-1, 1, (self.row * self.col, self.z_dim)) 
        # 連続的に潜在変数を変化させる際の開始、終了変数
        self.noise_fix2 = np.random.uniform(-1, 1, (1, self.z_dim))
        self.noise_fix3 = np.random.uniform(-1, 1, (1, self.z_dim))

        # 横軸がiteration数のプロット保存用np.ndarray
        self.g_loss_array = np.array([])
        self.d_loss_array = np.array([])
        self.d_accuracy_array = np.array([])
        self.d_predict_true_num_array = np.array([])
        self.c_predict_class_list = []

        #discriminator_optimizer = Adam(lr=1e-5, beta_1=0.1)
        combined_optimizer = Adam(lr=1e-4, beta_1=0.5, beta_2=0.9)

        # discriminatorモデル
        self.discriminator = self.build_discriminator()

        self.discriminator2 = self.build_discriminator2()
        
        # Generatorモデル
        self.generator, self.penalty = self.build_generator()

        

        # combinedモデルの学習時はdiscriminatorの学習をFalseにする
        for layer in self.discriminator.layers:
            layer.trainable = False
        self.discriminator.trainable = False
        for layer in self.discriminator2.layers:
            layer.trainable = False
        self.discriminator2.trainable = False

        self.netG_model, self.netG_train = self.build_combined()
        #形状の形成のdiscriminatorの学習をTrueにする
        for layer in self.discriminator.layers:
            layer.trainable = True
        for layer in self.generator.layers:
            layer.trainable = False
        self.discriminator.trainable = True
        self.generator.trainable = False
        self.netD_train = self.build_discriminator_with_own_loss()
       #接続用のdiscriminatorの学習をTrueにする
        for layer in self.discriminator.layers:
            layer.trainable = False
        self.discriminator.trainable = False
        for layer in self.discriminator2.layers:
            layer.trainable = True
        self.discriminator2.trainable = True
        self.netD_train2 = self.build_discriminator_with_own_loss2()
        

    def build_generator(self):
          latent = Input(shape=(self.z_dim, ))

          label1 = Input(shape = (self.output,1))
          label2 = Flatten()(label1)
          label3 = Dense(50, activation="sigmoid")(label2)

          con_latent = Concatenate(axis =1)([latent, label3])
          #print(latent1)
          dense1 = Dense(4*8*32*8)(con_latent)
          print(dense1)
          reshape = Reshape((8,64*16))(dense1)
          print(reshape)
          #norm1 = BatchNormalization(momentum=0.8)(reshape)
          acti1 = Activation("relu")(reshape)
          dec3_s = UpSampling1D(2)(acti1)#128,48       
          dec3 = Conv1D(64*4, 15, strides=1, padding ="same",activation = "relu")(dec3_s)#
          #norm2 = BatchNormalization(momentum=0.8)(dec3)
          #print(norm2)
          print(dec3)
          dec4_s = UpSampling1D(2)(dec3)#320,64
          dec4 = Conv1D(64*2, 15, strides=1, padding ="same",activation = "relu")(dec4_s)
          print(dec4)
          #norm3 = BatchNormalization(momentum=0.8)(dec4)
          #print(norm3)
          dec5_s = UpSampling1D(2)(dec4)
          dec5 = Conv1D(1, 15, strides=1, padding ="same",activation = "relu")(dec5_s)
          print(dec5)
          #norm4 = BatchNormalization(momentum=0.8)(dec5)
          #dec6_s = UpSampling1D(2)(dec5)
          #dec6 = Conv1D(1, 15, strides=1, padding ="same")(dec6_s)
          #acti2 = Activation("relu")(dec6)
          #flat = Flatten()(acti2)
          #dense2 = Dense(self.output)(flat)
          #print(dec6)
          #norm5 = BatchNormalization(momentum=0.8)(dec6)
          #dec7 = Dense(self.output)(dec6)
          #dec = Lambda(lambda x : backend.expand_dims(x, axis = 2))(dense2)
          out1 = Activation("tanh")(dec5)

          out2 = Concatenate(axis = 1)([label1, out1])
          
          model1 = Model([latent, label1], output = out1)
          model2 = Model([latent, label1], output = out2)
          return model1, model2

    def build_discriminator(self):
          
          img = (self.output,1)

          model = Sequential()

          model.add(Conv1D(64, kernel_size=25, strides = 4, input_shape = img, padding="same"))
          model.add(LeakyReLU(alpha=0.2))
          model.add(MaxPooling1D(2, padding='same'))
          model.add(Conv1D(128, kernel_size=25, strides = 4 , padding="same"))
          #model.add(BatchNormalization(momentum=0.8))
          model.add(LeakyReLU(alpha=0.2))
          model.add(MaxPooling1D(2, padding='same'))
          model.add(Conv1D(256, kernel_size=25, strides=4, padding="same"))
          model.add(LeakyReLU(alpha=0.2))
          model.add(MaxPooling1D(4, padding='same'))
          model.add(Conv1D(512, kernel_size=25, strides=4, padding="same"))
          model.add(LeakyReLU(alpha=0.2))
          model.add(Flatten())
          model.add(Dense(1))
          model.summary()
          return model

    def build_discriminator2(self):
          
          img = (self.output*2,1)

          model = Sequential()

          model.add(Conv1D(64, kernel_size=25, strides = 4, input_shape = img, padding="same"))
          model.add(LeakyReLU(alpha=0.2))
          model.add(MaxPooling1D(2, padding='same'))
          model.add(Conv1D(128, kernel_size=25, strides = 4 , padding="same"))
          #model.add(BatchNormalization(momentum=0.8))
          model.add(LeakyReLU(alpha=0.2))
          model.add(MaxPooling1D(2, padding='same'))
          model.add(Conv1D(256, kernel_size=25, strides=4, padding="same"))
          model.add(LeakyReLU(alpha=0.2))
          model.add(MaxPooling1D(4, padding='same'))
          model.add(Conv1D(512, kernel_size=25, strides=4, padding="same"))
          model.add(LeakyReLU(alpha=0.2))
          model.add(Flatten())
          model.add(Dense(1))
          model.summary()
          return model
    
    
    def build_combined1(self):
        z1 = Input(shape=(self.z_dim,))
        
        img = self.generator(z1)
        
        print("img",img)
        #img = Lambda(lambda x : backend.expand_dims(x, axis = 2))(img1)
        valid = self.discriminator(img)
        model = Model(z1, valid)
        model.summary()
        loss = -1. * K.mean(valid)
        training_updates = Adam(lr=1e-4, beta_1=0.5, beta_2=0.9).get_updates(self.generator.trainable_weights,[],loss)

        g_train = K.function([z1],\
                                [loss],    \
                                training_updates)

        return model, g_train
    def build_combined(self):
        z = Input(shape=(self.z_dim,))
        pre_input = Input(shape = (self.output,1))
        img1 = self.generator([z, pre_input])
        img2 = self.penalty([z, pre_input])
        valid1 = self.discriminator(img1)
        valid2 = self.discriminator2(img2)
        model = Model([z, pre_input], [valid1, valid2])
        model.summary()
        loss1 = -1. * K.mean(valid1)
        loss2 = -1. * K.mean(valid2)
        training_updates = Adam(lr=1e-4, beta_1=0.5, beta_2=0.9).get_updates(self.generator.trainable_weights,[],[loss1,loss2])
        loss = loss1 - loss2
        g_train = K.function([z, pre_input],\
                                [loss1,loss2],    \
                                training_updates)

        return model, g_train

    def build_discriminator_with_own_loss(self):

        ##モデルの定義
        # generatorの入力
        z1 = Input(shape=(self.z_dim,))
        pre_input=Input(shape = (self.output,1,))
        #z2 = Input(shape=(self.z_dim,))
        # discriimnatorの入力
        f_img = self.generator([z1, pre_input])
        
        img_shape = (self.output, 1)
        print("r:",img_shape)
        r_img = Input(shape=(img_shape))
        print(r_img)
        e_input = K.placeholder(shape=(None,1,1))
        a_img = Input(shape=(img_shape),\
        tensor=e_input * r_img + (1-e_input) * f_img)
        print(e_input)
        # discriminatorの出力
        f_out = self.discriminator(f_img)
        r_out = self.discriminator(r_img)
        a_out = self.discriminator(a_img)
        ##モデルの定義終了

        # 損失関数を定義する
        # original critic loss
        loss_real = K.mean(r_out) / BATCH_SIZE
        loss_fake = K.mean(f_out) / BATCH_SIZE

        # gradient penalty
        grad_mixed = K.gradients(a_out, [a_img])[0]
        norm_grad_mixed = K.sqrt(K.sum(K.square(grad_mixed), axis=[1,2]))
        grad_penalty = K.mean(K.square(norm_grad_mixed -1))

        # 最終的な損失関数
        loss = loss_fake - loss_real + GRADIENT_PENALTY_WEIGHT * grad_penalty

        # オプティマイザーと損失関数、学習する重みを指定する
        training_updates = Adam(lr=1e-4, beta_1=0.5, beta_2=0.9)\
                            .get_updates(self.discriminator.trainable_weights,[],loss)

        # 入出力とtraining_updatesをfunction化
        d_train = K.function([r_img, z1,pre_input, e_input],\
                                [loss_real, loss_fake],    \
                                 training_updates)
        return d_train

    def build_discriminator_with_own_loss2(self):

        ##モデルの定義
        # generatorの入力
        z1 = Input(shape=(self.z_dim,))
        pre_input = Input(shape =(self.output,1,))
        # discriimnatorの入力
        f_img = self.penalty([z1, pre_input])
        img_shape = (self.output*2, 1)
        print("r:",img_shape)
        r_img = Input(shape=(img_shape))
        print(r_img)
        e_input = K.placeholder(shape=(None,1,1))
        a_img = Input(shape=(img_shape),\
        tensor=e_input * r_img + (1-e_input) * f_img)
        print(e_input)
        # discriminatorの出力
        f_out = self.discriminator2(f_img)
        r_out = self.discriminator2(r_img)
        a_out = self.discriminator2(a_img)
        ##モデルの定義終了

        # 損失関数を定義する
        # original critic loss
        loss_real = K.mean(r_out) / BATCH_SIZE
        loss_fake = K.mean(f_out) / BATCH_SIZE
        # gradient penalty
        grad_mixed = K.gradients(a_out, [a_img])[0]
        norm_grad_mixed = K.sqrt(K.sum(K.square(grad_mixed), axis=[1,2]))
        grad_penalty = K.mean(K.square(norm_grad_mixed -1))
        # 最終的な損失関数
        loss = loss_fake - loss_real + GRADIENT_PENALTY_WEIGHT * grad_penalty
        # オプティマイザーと損失関数、学習する重みを指定する
        training_updates = Adam(lr=1e-4, beta_1=0.5, beta_2=0.9)\
                            .get_updates(self.discriminator2.trainable_weights,[],loss)
        # 入出力とtraining_updatesをfunction化
        d_train = K.function([r_img, z1, pre_input, e_input],\
                                [loss_real, loss_fake],    \
                                 training_updates)
        return d_train

    def sin(self,x,T=96):
          return np.sin(2*np.pi*x/T)+2.0*np.sin(np.pi*x/(2*T))
    def toy_problem(self,T=10000, ampl = 1):
          x = np.arange(0, 2*T+1)
          noise = ampl*np.random.uniform(low=-0.01, high = 0.01, size=len(x))
          return self.sin(x) + noise
    def make_dataset(self, rdata, n ,n_prev = 100):
           data, target =[], []
           maxlen = n
           if rdata.ndim == 1:
               for i in range(len(rdata)-maxlen):
                   data.append(rdata[i:i+maxlen])
           else:
               for m in range(len(rdata)):
                   for i in range(len(rdata[m])-maxlen):
                       data.append(rdata[m][i:i+maxlen])
           re_data = np.array(data).reshape(len(data),maxlen,1)
           return re_data           
    
    def min_max(self, x, axis=None):
        min = x.min(axis=axis, keepdims=True)
        max = x.max(axis=axis, keepdims=True)
        result = (x-min)/(max-min)
        return result
    def load_weights(self, g_weight, d_weight, en_weight):
        self.generator.load_weights(g_weight)
        self.discriminator.load_weights(d_weight)
        self.discriminator2.load_weights(en_weight)
    def train(self,data,epochs=100,batch_size=128,save_interval=50):
        #f = self.toy_problem()  
        X_train = self.make_dataset(data, n=64)
        X_train2 = self.make_dataset(data, n = 128)

        #) 値を-1 to 1に規格化
        X_train = self.min_max(X_train)
        X_train2 = self.min_max(X_train2)
        self.load_weights("rmsd/weights/generator_128_600000.h5","rmsd/weights/discriminator_128_600000.h5", "rmsd/weights/discriminator2_128_600000.h5")
        self.g_loss_array = np.zeros(epochs)
        self.d_loss_array = np.zeros(epochs)
        self.d_accuracy_array = np.zeros(epochs)
        self.d_predict_true_num_array = np.zeros(epochs)
        # gradient_penalty loss function and is not used.
        positive_y = np.ones((batch_size, 1), dtype=np.float32)
        negative_y = -positive_y
        dummy_y = np.zeros((batch_size, 1), dtype=np.float32)
        g_loss_hist=[]
        d_loss_hist=[]
        for epoch in range(epochs):
            epoch = epoch+600000
            for j in range(TRAINING_RATIO):

#時系列データの学習
                # ---------------------
                #  Discriminatorの学習
                # ---------------------
    
                # バッチサイズをGeneratorから生成
                noise = np.random.uniform(-1, 1, (batch_size, self.z_dim))
                
                #gen_imgs = np.expand_dims(gen_imgs, axis=2)
                
                # バッチサイズを教師データからピックアップ
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]
                
                gen_imgs = self.generator.predict([noise, imgs])
                epsilon = np.random.uniform(size = (batch_size, 1,1))
                errD_real, errD_fake = self.netD_train([imgs, noise, imgs ,epsilon])
                d_loss = errD_real - errD_fake
                # discriminatorを学習
                # 本物データと偽物データは一緒に学習させる
                noise2 = np.random.uniform(-1, 1, (batch_size, self.z_dim))
                idx_ = np.random.randint(0, X_train2.shape[0], batch_size)
                imgs2 = X_train2[idx_]
                epsilon2 = np.random.uniform(size = (batch_size, 1,1))
                errD_real2, errD_fake2 = self.netD_train2([imgs2,  noise2, imgs, epsilon2])
                d_loss2 = errD_real2 - errD_fake2


                # discriminatorの予測（本物と偽物が半々のミニバッチ）
                d_predict = self.discriminator.predict_classes(np.concatenate([gen_imgs,imgs]), verbose=0)
                d_predict = np.sum(d_predict)

            #c_predict = self.classifier.predict_classes(np.concatenate([gen_imgs,imgs]), verbose=0)

#時系列データの学習
            # ---------------------
            #  Generatorの学習
            # ---------------------

            noise = np.random.uniform(-1, 1, (batch_size, self.z_dim))


            # 生成データの正解ラベルは本物（1） 
            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.netG_train([noise, imgs])
            #接続用のGANの学習
            
            g_loss_hist.append(g_loss[0])
            d_loss_hist.append(d_loss)
            # 進捗の表示
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss, g_loss[0]))

            # np.ndarrayにloss関数を格納
            #self.g_loss_array[epoch] = g_loss[0]
            #self.d_loss_array[epoch] = d_loss
            #self.d_accuracy_array[epoch] = d_loss
            #self.d_predict_true_num_array[epoch] = d_predict
            #self.c_predict_class_list.append(c_predict)

            if epoch % save_interval == 0:
                #重みを保存する
                #g_weight = self.path+'weights/generator_' + str(epoch) + '.h5'
                #self.generator.save_weights(g_weight, True)
                #d_weight = self.path+'weights/discriminator_' + str(epoch) + '.h5'
                #self.discriminator.save_weights(d_weight, True)
                # 毎回異なる乱数から画像を生成
#                self.save_imgs(self.row, self.col, epoch, '', noise, imgs)
                self.generator.save_weights("rmsd/weights/generator_128_%s.h5" % epoch)
                self.discriminator.save_weights("rmsd/weights/discriminator_128_%s.h5" % epoch)
                self.discriminator2.save_weights("rmsd/weights/discriminator2_128_%s.h5" % epoch)
                # 学習結果をプロット
#                fig, ax = plt.subplots(4,1, figsize=(8.27,11.69))
#                ax[0].plot(self.g_loss_array[:epoch])
#                ax[0].set_title("g_loss")
#                ax[1].plot(self.d_loss_array[:epoch])
#                ax[1].set_title("d_loss")
#                ax[2].plot(self.d_accuracy_array[:epoch])
#                ax[2].set_title("d_accuracy")
#                ax[3].plot(self.d_predict_true_num_array[:epoch])
#                ax[3].set_title("d_predict_true_num_array")
#                fig.suptitle("epoch: %5d" % epoch)
#                fig.savefig(self.path + "training_%d.png" % epoch)
#                plt.close()
            np.save("rmsd/g_loss_600000.npy",g_loss_hist)
            np.save("rmsd/d_loss_600000.npy",d_loss_hist)
        # 重みを保存
    def save_imgs(self, row, col, epoch, filename, noise, dataset):
        # row, col
        # 生成画像を敷き詰めるときの行数、列数
        
        #idx = np.random.randint(0, dataset.shape[0], batch_size)
        gen_img = self.generator.predict([noise, dataset])
        print(dataset.shape, gen_img.shape)
        a = np.concatenate([dataset, gen_img], axis=1)
        imgs = a
        
        # 生成画像を0-1に再スケール
        #gen_imgs = 0.5 * gen_imgs + 0.5
        fig = plt.figure()
        ax1 = fig.add_subplot(4, 4, 1) 
        ax1.plot(imgs[0]) 
        ax2 = fig.add_subplot(4, 4, 2)  
        ax2.plot(imgs[1])
        ax3 = fig.add_subplot(4, 4, 3)  
        ax3.plot(imgs[2])
        ax4 = fig.add_subplot(4, 4, 4)  
        ax4.plot(imgs[3])
        ax5 = fig.add_subplot(4, 4, 5)  
        ax5.plot(imgs[4])
        ax6 = fig.add_subplot(4, 4, 6)  
        ax6.plot(imgs[5])
        ax7 = fig.add_subplot(4, 4, 7)  
        ax7.plot(imgs[6])
        ax8 = fig.add_subplot(4, 4, 8)  
        ax8.plot(imgs[7])
        ax9 = fig.add_subplot(4, 4, 9)  
        ax9.plot(imgs[8])
        ax10 = fig.add_subplot(4, 4, 10)  
        ax10.plot(imgs[9])
        ax11 = fig.add_subplot(4, 4, 11)  
        ax11.plot(imgs[10])
        ax12 = fig.add_subplot(4, 4, 12)  
        ax12.plot(imgs[11])
        ax13 = fig.add_subplot(4, 4, 13)  
        ax13.plot(imgs[12])
        ax14 = fig.add_subplot(4, 4, 14)  
        ax14.plot(imgs[13])
        ax15 = fig.add_subplot(4, 4, 15)  
        ax15.plot(imgs[14])
        ax16 = fig.add_subplot(4, 4, 16)  
        ax16.plot(imgs[15])
	# save plot to file
        filename1 = drive_root_dir+'plot_%04d.png' %epoch
        plt.savefig(filename1)
        plt.close()

if __name__ == '__main__':
    from collect_traj import collect
    path_traj = sys.argv[1]
    path_gro = sys.argv[2]
    ref = sys.argv[3]
    X_train = collect(path_traj, path_gro, ref)
    print(X_train.shape) 
    gan = GAN_GP()
    gan.train(X_train,epochs=300001,batch_size=BATCH_SIZE,save_interval=50000)

