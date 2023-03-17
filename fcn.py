from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D, Conv2DTranspose, Add, Dropout
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from keras import layers
import numpy as np


class FCNModel(object):

    def __init__(self):
    	print("FCN Model initialized")


    def identity_block(self, input_tensor, kernel_size, filters, stage, block):
       
        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        act_name = 'act' + str(stage)+ block

        x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2, kernel_size,
                padding='same', name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        x = layers.add([x, input_tensor])
        x = Activation('relu', name=act_name)(x)
        return x


    def conv_block(self, input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    
        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        act_name = 'act' + str(stage) + block

        x = Conv2D(filters1, (1, 1), strides=strides,
                name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2, kernel_size, padding='same',
                name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        shortcut = Conv2D(filters3, (1, 1), strides=strides,
                        name=conv_name_base + '1')(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

        x = layers.add([x, shortcut])
        x = Activation('relu', name=act_name)(x)
        return x


    def build_fcn(self, input_shape, num_classes):    
        inputs = Input(input_shape)
        N_CLASSES = num_classes
        bn_axis = 3

        x = ZeroPadding2D((3, 3))(inputs)
        x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
        x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = self.conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='c')

        x = self.conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        c3 = self.identity_block(x, 3, [128, 128, 512], stage=3, block='d')

        x = self.conv_block(c3, 3, [256, 256, 1024], stage=4, block='a')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
        c4 = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

        x = self.conv_block(c4, 3, [512, 512, 2048], stage=5, block='a')
        x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        c5 = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
        
    #     conv_p1 = Conv2D(2048, (7, 7), strides=(1, 1), padding='valid', kernel_initializer='he_normal')(c5)
    #     drop_p1 = Dropout(0.5)(conv_p1)
        conv_p1 = Conv2D(2048, (1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal')(c5)
        drop_p1 = Dropout(0.5)(conv_p1)
        
        score_c5 = Conv2D(N_CLASSES, (1, 1), strides=(1, 1), padding='same', kernel_initializer='zeros')(drop_p1)
        up_c5 = Conv2DTranspose(N_CLASSES, (2, 2), strides=(2, 2), padding='valid')(score_c5)
        
        score_c4 = Conv2D(N_CLASSES, (1, 1), strides=(1, 1), padding='same', kernel_initializer='zeros')(c4)
        fuse_16 = Add()([score_c4, up_c5])
        up_c4 = Conv2DTranspose(N_CLASSES, (2, 2), strides=(2, 2), padding='valid')(fuse_16)
        
        score_c3 = Conv2D(N_CLASSES, (1, 1), strides=(1, 1), padding='same', kernel_initializer='zeros')(c3)
        fuse_32 = Add()([score_c3, up_c4])
        up_c3 = Conv2DTranspose(N_CLASSES, (8, 8), strides=(8, 8), padding='valid', activation='sigmoid')(fuse_32)

        fcn_model = Model(inputs=inputs, outputs=up_c3)        
        return fcn_model
