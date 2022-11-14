# -*- coding: utf-8 -*-

from google.colab import drive
drive.mount('/content/drive')

!pip install keras --user

# installing kaggle library to import the data directly into Colab notebook
 ! pip install -q kaggle

from google.colab import files
uploaded = files.upload()

! mkdir ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json

! kaggle datasets download -d balraj98/massachusetts-roads-dataset

# unzipping the files
! unzip massachusetts-roads-dataset.zip -d massachusetts-roads-dataset &> /dev/null

# Commented out IPython magic to ensure Python compatibility.
from tensorflow.python.keras import callbacks, optimizers
from tensorflow.python.keras.models import Model, load_model, model_from_json
from keras.preprocessing import image
from keras.layers import Input, Conv2D, MaxPooling2D, Activation, BatchNormalization, Conv2DTranspose, Concatenate
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
#from keras.layers import Conv2D, Conv2DTranspose, Input, MaxPooling2D, Add, BatchNormalization, LeakyReLU, Reshape, Flatten, Dense,PReLU,add
#from tensorflow.keras.optimizers import Adam
 
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from keras import backend as K
 
import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
# %matplotlib inline
 
K.set_image_data_format('channels_last')

train_dir = '/content/massachusetts-roads-dataset/tiff/train/'
mask_dir = '/content/massachusetts-roads-dataset/tiff/train_labels/'
 
val_dir = '/content/massachusetts-roads-dataset/tiff/val/'
v_mask_dir = '/content/massachusetts-roads-dataset/tiff/val_labels/'

test_dir = '/content/massachusetts-roads-dataset/tiff/test/'
t_mask_dir = '/content/massachusetts-roads-dataset/tiff/test_labels/'

image_shape = (256,256)

def preprocess_mask_image2(image, class_num, color_limit):
  pic = np.array(image)
  img = np.zeros((pic.shape[0], pic.shape[1], 1))  
  np.place(img[ :, :, 0], pic[ :, :, 0] >= color_limit, 1)  
  return img

def dice_coef(y_true, y_pred):

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0) 
 
def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def train_generator(img_dir, label_dir, batch_size, input_size):
    list_images = os.listdir(label_dir)
    # shuffle(list_images) #Randomize the choice of batches
    ids_train_split = range(len(list_images))

    while True:
         for start in range(0, len(ids_train_split), batch_size):
            x_batch = []
            y_batch = []

            end = min(start + batch_size, len(ids_train_split))
            ids_train_batch = ids_train_split[start:end]

            for id in ids_train_batch:
              img_name = img_dir + list_images[id]+'f'
              mask_name = label_dir + list_images[id]
  
              img = cv2.imread(img_name) 
              img  = cv2.resize(img, image_shape, interpolation=cv2.INTER_AREA)
  
              mask = cv2.imread(mask_name)
              mask = cv2.resize(mask, image_shape, interpolation=cv2.INTER_AREA)
              mask = preprocess_mask_image2(mask, 2, 50)                
              
              x_batch += [img]
              y_batch += [mask]    

    
            x_batch = np.array(x_batch) / 255.
            y_batch = np.array(y_batch) 

            yield x_batch, np.expand_dims(y_batch, -1)

from tensorflow.keras.optimizers import Adam

import keras

def unet(num_classes = 1, input_shape= (image_shape[0],image_shape[1], 3)):
  inp = Input(input_shape)
  # Block 1
  x = Conv2D(64, (3, 3), padding='same', name='block1_conv1')(inp)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(64, (3, 3), padding='same', name='block1_conv2')(x)
  x = BatchNormalization()(x)
  block_1_out = Activation('relu')(x)
  x = MaxPooling2D()(block_1_out)
  # Block 2
  x = Conv2D(128, (3, 3), padding='same', name='block2_conv1')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(128, (3, 3), padding='same', name='block2_conv2')(x)
  x = BatchNormalization()(x)
  block_2_out = Activation('relu')(x)
  x = MaxPooling2D()(block_2_out)
  # Block 3
  x = Conv2D(256, (3, 3), padding='same', name='block3_conv1')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(256, (3, 3), padding='same', name='block3_conv2')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(256, (3, 3), padding='same', name='block3_conv3')(x)
  x = BatchNormalization()(x)
  block_3_out = Activation('relu')(x)
  x = MaxPooling2D()(block_3_out)
  # Block 4
  x = Conv2D(512, (3, 3), padding='same', name='block4_conv1')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(512, (3, 3), padding='same', name='block4_conv2')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(512, (3, 3), padding='same', name='block4_conv3')(x)
  x = BatchNormalization()(x)
  block_4_out = Activation('relu')(x)
  
 
  
  x = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', name = 'Conv2DTranspose_UP2')(block_4_out)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Concatenate()([x, block_3_out])
  x = Conv2D(256, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(256, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  # UP 3
  x = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', name = 'Conv2DTranspose_UP3')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Concatenate()([x, block_2_out])
  x = Conv2D(128, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(128, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  # UP 4
  x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', name = 'Conv2DTranspose_UP4')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Concatenate()([x, block_1_out])
  x = Conv2D(64, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(64, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(num_classes, (3, 3), activation='sigmoid', padding='same')(x)
 
  model = Model(inputs=inp, outputs=x)
  opt = tf.keras.optimizers.Adam#(lr=0.0001)
  model.compile(optimizer='adam',
                   loss=dice_coef_loss,
                   metrics=[dice_coef])
  # model.compile(optimizer= 'adam' , loss= keras.losses.binary_crossentropy, metrics=['accuracy'])

    # model.summary()
  return model

model = unet()
model.summary()

batch_size = 1
history = model.fit_generator(train_generator(train_dir, mask_dir, batch_size, image_shape),                              
                              steps_per_epoch=554,
                              epochs=2,
                              verbose=1,
                              # callbacks=callbacks,
                              validation_data=train_generator(val_dir, v_mask_dir, batch_size, image_shape),
                              validation_steps=1,
                              class_weight=None,
                              max_queue_size=10,
                              workers=1
                              )

model.save_weights('/content/drive/MyDrive/Internship/weights/')

model1 = unet()

model1.load_weights('/content/drive/MyDrive/Internship/weights/')

def prepare_test_image(image):    
  x_batch = []   
  # img = cv2.imread(image_path)  
  img  = cv2.resize(image, image_shape, interpolation=cv2.INTER_AREA)
  x_batch += [img]           
  x_batch = np.array(x_batch) / 255.        

  return x_batch

def binaryImage(image):
    x = image.shape[1]
    y = image.shape[2]
    imgs = np.zeros((x,y,3))
    for k in range(x):
        for n in range(y):
            if image[0,k,n]>0.5:
                imgs[k,n,0] = 255
                imgs[k,n,1] = 255
                imgs[k,n,2] = 255
    return imgs

def draw(orig_im, mask_im,recogn_im,out_im):
    plt.figure(figsize=(20,17))
    plt.subplot(1,3,1)
    plt.title('Original')
    plt.imshow(orig_im)
    plt.subplot(1,3,3)
    plt.title('Mask Original')
    plt.imshow(mask_im)
    plt.subplot(1,3,2)
    plt.title('Recogn Roads')
    plt.imshow(recogn_im)
    plt.axis('off')
    plt.show()




def recogn_test_image():
    test_images = os.listdir(t_mask_dir)
    
    for test in test_images:
        im_test = cv2.imread(test_dir+test+'f')
        im_mask = cv2.imread(t_mask_dir+test)
        out_test = model.predict(prepare_test_image(im_test), verbose=0)
        img_r = binaryImage(out_test)
        draw(im_test, im_mask, img_r, out_test[0,:, :, 0]*255)

preds= model.predict(x_batch[0:4,:,:,:])

recogn_test_image()
