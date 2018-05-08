import tensorflow as tf


class TransformLayer(object):
    """
    Transformation layer
    BatchNorm -- Scale -- ReLU -- Conv -- Dropout -- Concat
    """

    def __init__(self, growth_rate=12, dropout_rate=0.2):
        init = tf.truncated_normal_initializer(stddev=0.01)
        self.bnorm = tf.layers.BatchNormalization()
        self.conv = tf.layers.Conv3D(filters=growth_rate, kernel_size=3, padding="same", kernel_initializer=init)
        self.dropout = tf.layers.Dropout(rate=dropout_rate)

    def __call__(self, x, train):
        h = tf.nn.relu(self.bnorm(x, training=train))
        h = self.conv(h)
        h = self.dropout(h)
        return tf.concat([h, x], axis=4)


class DenseBlock(object):
    """
    DenseBlock: transformation layer * 12
    """

    def __init__(self, n_transform_layer=12):
        self.n_transform_layer = n_transform_layer
        self.transformlayer1 = TransformLayer()
        self.transformlayer2 = TransformLayer()
        self.transformlayer3 = TransformLayer()
        self.transformlayer4 = TransformLayer()
        self.transformlayer5 = TransformLayer()
        self.transformlayer6 = TransformLayer()
        self.transformlayer7 = TransformLayer()
        self.transformlayer8 = TransformLayer()
        self.transformlayer9 = TransformLayer()
        self.transformlayer10 = TransformLayer()
        self.transformlayer11 = TransformLayer()
        self.transformlayer12 = TransformLayer()

    def __call__(self, x, train):
        h = self.transformlayer1(x, train=train)
        h = self.transformlayer2(h, train=train)
        h = self.transformlayer3(h, train=train)
        h = self.transformlayer4(h, train=train)
        h = self.transformlayer5(h, train=train)
        h = self.transformlayer6(h, train=train)
        h = self.transformlayer7(h, train=train)
        h = self.transformlayer8(h, train=train)
        h = self.transformlayer9(h, train=train)
        h = self.transformlayer10(h, train=train)
        h = self.transformlayer11(h, train=train)
        h = self.transformlayer12(h, train=train)

        return h


class DenseVoxNet(object):
    """Voxel Densely Network"""

    def __init__(self, n_classes=3, dropout_rate=0.2, is_training=True):
        init = tf.truncated_normal_initializer(stddev=0.01)
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate
        self.is_training = is_training
        self.conv1 = tf.layers.Conv3D(filters=16, kernel_size=3, strides=2, padding="same", kernel_initializer=init)
        self.denseblock1 = DenseBlock()
        self.bnorm13 = tf.layers.BatchNormalization()
        self.conv14 = tf.layers.Conv3D(filters=160, kernel_size=1, padding="valid", kernel_initializer=init)
        self.dropout13 = tf.layers.Dropout(rate=self.dropout_rate)
        self.deconv14 = tf.layers.Conv3DTranspose(filters=64, kernel_size=4, strides=2, padding="same", use_bias=False,
                                                  kernel_initializer=init)
        self.conv_out1 = tf.layers.Conv3D(filters=self.n_classes, kernel_size=1, padding="same",
                                          kernel_initializer=init)
        self.pooling1 = tf.layers.MaxPooling3D(pool_size=2, strides=2)
        self.denseblock2 = DenseBlock()
        self.bnorm26 = tf.layers.BatchNormalization()
        self.conv27 = tf.layers.Conv3D(filters=304, kernel_size=1, padding="valid", kernel_initializer=init)
        self.deconv27_1 = tf.layers.Conv3DTranspose(filters=128, kernel_size=4, strides=2, padding="same",
                                                    use_bias=False, kernel_initializer=init)
        self.deconv27_2 = tf.layers.Conv3DTranspose(filters=64, kernel_size=4, strides=2, padding="same",
                                                    use_bias=False, kernel_initializer=init)
        self.conv_out2 = tf.layers.Conv3D(filters=self.n_classes, kernel_size=1, padding="same",
                                          kernel_initializer=init)

    def __call__(self, x):
        """
        calculate output of DenseNet given input x
        :param x: (batch_size, xlen, ylen, zlen, in_channels)
        :param train:
        :return: (batch_size, xlen, ylen, zlen, n_classess)
        """
        h = self.conv1(x)
        h = self.denseblock1(h, train=self.is_training)
        h = tf.nn.relu(self.bnorm13(h, training=self.is_training))
        h = self.dropout13(self.conv14(h))
        deconv14 = tf.nn.relu(self.deconv14(h))
        logits1 = self.conv_out1(deconv14)

        h = self.pooling1(h)
        h = self.denseblock2(h, train=self.is_training)
        h = tf.nn.relu(self.bnorm26(h, training=self.is_training))
        h = self.conv27(h)
        h = tf.nn.relu(self.deconv27_1(h))
        h = tf.nn.relu(self.deconv27_2(h))
        logits2 = self.conv_out2(h)

        prob_map = tf.nn.softmax(logits2, dim=4)
        annotation_pred = tf.argmax(logits2, axis=4, name="prediction")

        return logits1, logits2, prob_map, tf.expand_dims(annotation_pred, dim=4)
