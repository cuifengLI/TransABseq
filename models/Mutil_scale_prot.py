import tensorflow as tf
from tensorflow.keras import layers


class MultiScaleConvA(layers.Layer):
    def __init__(self):
        super(MultiScaleConvA, self).__init__()
        self.conv_11 = layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu')
        self.bn_11 = layers.BatchNormalization()
        self.maxpool_11 = layers.MaxPooling2D(pool_size=(5, 5), strides=(2, 2), padding='valid')
        self.conv_12 = layers.Conv2D(128, (5, 5), strides=(1, 1), padding='same', activation='relu')
        self.bn_12 = layers.BatchNormalization()
        self.maxpool_12 = layers.MaxPooling2D(pool_size=(5, 5), strides=(2, 2), padding='valid')
        self.conv_13 = layers.Conv2D(128, (7, 7), strides=(1, 1), padding='same', activation='relu')
        self.bn_13 = layers.BatchNormalization()
        self.maxpool_13 = layers.MaxPooling2D(pool_size=(5, 5), strides=(2, 2), padding='valid')
        self.add1 = layers.Add()
        self.bn1 = layers.BatchNormalization()

        self.conv_21 = layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')
        self.bn_21 = layers.BatchNormalization()
        self.maxpool_21 = layers.MaxPooling2D(pool_size=(5, 5), strides=(2, 2), padding='valid')
        self.conv_22 = layers.Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu')
        self.bn_22 = layers.BatchNormalization()
        self.maxpool_22 = layers.MaxPooling2D(pool_size=(5, 5), strides=(2, 2), padding='valid')
        self.conv_23 = layers.Conv2D(256, (7, 7), strides=(1, 1), padding='same', activation='relu')
        self.bn_23 = layers.BatchNormalization()
        self.maxpool_23 = layers.MaxPooling2D(pool_size=(5, 5), strides=(2, 2), padding='valid')
        self.add2 = layers.Add()
        self.bn2 = layers.BatchNormalization()

        self.conv_31 = layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')
        self.bn_31 = layers.BatchNormalization()
        self.maxpool_31 = layers.MaxPooling2D(pool_size=(5, 5), strides=(2, 2), padding='valid')
        self.conv_32 = layers.Conv2D(512, (5, 5), strides=(1, 1), padding='same', activation='relu')
        self.bn_32 = layers.BatchNormalization()
        self.maxpool_32 = layers.MaxPooling2D(pool_size=(5, 5), strides=(2, 2), padding='valid')
        self.conv_33 = layers.Conv2D(512, (7, 7), strides=(1, 1), padding='same', activation='relu')
        self.bn_33 = layers.BatchNormalization()
        self.maxpool_33 = layers.MaxPooling2D(pool_size=(5, 5), strides=(2, 2), padding='valid')
        self.add3 = layers.Add()
        self.bn3 = layers.BatchNormalization()
        self.flat = layers.Flatten()

    def call(self, inputs, training=True, mask=None):
        inputs = tf.expand_dims(inputs, axis=-1)
        conv11 = self.conv_11(inputs)
        bn11 = self.bn_11(conv11)
        maxpool11 = self.maxpool_11(bn11)
        conv12 = self.conv_12(inputs)
        bn12 = self.bn_12(conv12)
        maxpool12 = self.maxpool_12(bn12)
        conv13 = self.conv_13(inputs)
        bn13 = self.bn_13(conv13)
        maxpool13 = self.maxpool_13(bn13)
        Add1 = self.add1([maxpool11, maxpool12, maxpool13])
        BN1 = self.bn1(Add1)

        conv21 = self.conv_21(BN1)
        bn21 = self.bn_21(conv21)
        maxpool21 = self.maxpool_21(bn21)
        conv22 = self.conv_22(BN1)
        bn22 = self.bn_22(conv22)
        maxpool22 = self.maxpool_22(bn22)
        conv23 = self.conv_23(BN1)
        bn23 = self.bn_23(conv23)
        maxpool23 = self.maxpool_23(bn23)
        Add2 = self.add2([maxpool21, maxpool22, maxpool23])
        BN2 = self.bn2(Add2)

        conv31 = self.conv_31(BN2)
        bn31 = self.bn_31(conv31)
        maxpool31 = self.maxpool_31(bn31)
        conv32 = self.conv_32(BN2)
        bn32 = self.bn_32(conv32)
        maxpool32 = self.maxpool_32(bn32)
        conv33 = self.conv_33(BN2)
        bn33 = self.bn_33(conv33)
        maxpool33 = self.maxpool_33(bn33)
        Add3 = self.add3([maxpool31, maxpool32, maxpool33])
        BN3 = self.bn3(Add3)

#         print(BN3.shape[1])
        BN3 = adaptive_arg_pool2d(BN3, BN3.shape[1], BN3.shape[2])
        flat = self.flat(BN3)
#         print(flat.shape)

        return flat


def adaptive_arg_pool2d(input, height_size, weight_size):
    arg_pool = layers.AveragePooling2D(pool_size=(height_size, weight_size), strides=(1, 1), padding='valid')
    output = arg_pool(input)
    return output



