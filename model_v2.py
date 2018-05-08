import tensorflow as tf


class DenseVoxNet:
    def __init__(self, n_classes=3, dropout_rate=0.2, is_training=True):
        self.init = tf.truncated_normal_initializer(stddev=0.01)
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate
        self.is_training = is_training

    def __call__(self, x):
        with tf.name_scope("conv1"):
            conv1 = tf.layers.conv3d(x, filters=16, kernel_size=3, strides=2, padding="same",
                                     kernel_initializer=self.init)
        with tf.name_scope("concat12"):
            concat12 = self.denseBlock(conv1)
        with tf.name_scope("pooling1"):
            conv14, pooling1 = self.transformLayer(concat12, n_filters=160)
        with tf.name_scope("concat24"):
            concat24 = self.denseBlock(pooling1)
        with tf.name_scope("bn26"):
            bn26 = tf.nn.relu(self.batch_normalization(concat24, training=self.is_training))
        with tf.name_scope("conv27"):
            conv27 = tf.layers.conv3d(bn26, filters=304, kernel_size=1, padding="valid", kernel_initializer=self.init)
        with tf.name_scope("deconv1"):
            deconv1 = tf.nn.relu(
                tf.layers.conv3d_transpose(conv27, filters=128, kernel_size=4, strides=2, use_bias=False,
                                           padding="same",
                                           kernel_initializer=self.init))
        with tf.name_scope("deconv2"):
            deconv2 = tf.nn.relu(
                tf.layers.conv3d_transpose(deconv1, filters=64, kernel_size=4, strides=2, use_bias=False,
                                           padding="same",
                                           kernel_initializer=self.init))
        with tf.name_scope("logit1"):
            logit1 = tf.layers.conv3d(deconv2, filters=self.n_classes, kernel_size=1, padding="same",
                                      kernel_initializer=self.init)

        with tf.name_scope("score_aux1"):
            socre_aux1 = tf.layers.conv3d(conv14, filters=self.n_classes, kernel_size=1, padding="same",
                                          kernel_initializer=self.init)

        with tf.name_scope("logit2"):
            logit2 = tf.layers.conv3d_transpose(socre_aux1, filters=self.n_classes, kernel_size=4, strides=2,
                                                use_bias=False,
                                                padding="same", kernel_initializer=self.init)

        prob_map = tf.nn.softmax(logit1, axis=4)
        annotation = tf.argmax(logit1, axis=4, name="prediction")

        return logit1, logit2, prob_map, annotation

    def denseBlock(self, x, n_layers=12):
        for _ in range(n_layers):
            x = self.unitLayer(x)

        return x

    def unitLayer(self, x, growth_rate=12):
        bn = tf.nn.relu(self.batch_normalization(x, training=self.is_training))
        conv = tf.layers.conv3d(bn, filters=growth_rate, kernel_size=3, padding="same", kernel_initializer=self.init)
        drop = tf.layers.dropout(conv, rate=self.dropout_rate, training=self.is_training)

        return tf.concat([drop, x], axis=4)

    def transformLayer(self, x, n_filters):
        bn = tf.nn.relu(self.batch_normalization(x, training=self.is_training))
        conv = tf.layers.conv3d(bn, filters=n_filters, kernel_size=1, padding="valid", kernel_initializer=self.init)
        drop = tf.layers.dropout(conv, rate=self.dropout_rate, training=self.is_training)
        pool = tf.layers.max_pooling3d(drop, pool_size=2, strides=2)

        return conv, pool

    def batch_normalization(self, x, training):
        training = tf.constant(training)
        depth = x.get_shape()[-1]
        beta = tf.Variable(tf.constant(0.0, shape=[depth]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[depth]), name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2, 3], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.99)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(training,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed_tensor = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        return normed_tensor
