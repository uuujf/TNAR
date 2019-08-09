import math
import numpy as np
import tensorflow as tf

def bn(x, dim, is_training=True, update_batch_stats=True, collections=None, name="bn"):
    params_shape = (dim,)
    n = tf.to_float(tf.reduce_prod(tf.shape(x)[:-1]))
    axis = list(range(int(tf.shape(x).get_shape().as_list()[0]) - 1))
    mean = tf.reduce_mean(x, axis)
    var = tf.reduce_mean(tf.pow(x - mean, 2.0), axis)
    avg_mean = tf.get_variable(
        name=name + "_mean",
        shape=params_shape,
        initializer=tf.constant_initializer(0.0),
        collections=collections,
        trainable=False
    )

    avg_var = tf.get_variable(
        name=name + "_var",
        shape=params_shape,
        initializer=tf.constant_initializer(1.0),
        collections=collections,
        trainable=False
    )

    gamma = tf.get_variable(
        name=name + "_gamma",
        shape=params_shape,
        initializer=tf.constant_initializer(1.0),
        collections=collections
    )

    beta = tf.get_variable(
        name=name + "_beta",
        shape=params_shape,
        initializer=tf.constant_initializer(0.0),
        collections=collections,
    )

    decay_factor = 0.99
    if is_training:
        avg_mean_assign_op = tf.no_op()
        avg_var_assign_op = tf.no_op()
        if update_batch_stats:
            avg_mean_assign_op = tf.assign(
                avg_mean,
                decay_factor * avg_mean + (1 - decay_factor) * mean)
            avg_var_assign_op = tf.assign(
                avg_var,
                decay_factor * avg_var + (n / (n - 1))
                * (1 - decay_factor) * var)

        with tf.control_dependencies([avg_mean_assign_op, avg_var_assign_op]):
            z = (x - mean) / tf.sqrt(1e-6 + var)
    else:
        z = (x - avg_mean) / tf.sqrt(1e-6 + avg_var)

    return gamma * z + beta

class VAE(object):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.reuse = {}
        self.latent_dim = latent_dim
        self.leaky = 0.1

    def encode(self, x, is_training, name='encoder'):
        if name in self.reuse.keys():
            reuse = self.reuse[name]
        else:
            self.reuse[name] = True
            reuse = False

        with tf.variable_scope(name, reuse=reuse) as scope:
            w1 = tf.get_variable(name='w_conv1', shape=[4,4,3,128], dtype=tf.float32)
            w2 = tf.get_variable(name='w_conv2', shape=[4,4,128,256], dtype=tf.float32)
            w3 = tf.get_variable(name='w_conv3', shape=[4,4,256,512], dtype=tf.float32)
            w21 = tf.get_variable(name='w_fc21', shape=[16*512,self.latent_dim], dtype=tf.float32)
            b21 = tf.get_variable(name='b_fc21', shape=[self.latent_dim], dtype=tf.float32)
            w22 = tf.get_variable(name='w_fc22', shape=[16*512, self.latent_dim], dtype=tf.float32)
            b22 = tf.get_variable(name='b_fc22', shape=[self.latent_dim], dtype=tf.float32)

            x = tf.nn.conv2d(x, w1, strides=[1,2,2,1], padding='SAME')
            x = bn(x, 128, is_training, is_training, name='bn1')
            x = tf.nn.leaky_relu(x, self.leaky)
            x = tf.nn.conv2d(x, w2, strides=[1,2,2,1], padding='SAME')
            x = bn(x, 256, is_training, is_training, name='bn2')
            x = tf.nn.leaky_relu(x, self.leaky)
            x = tf.nn.conv2d(x, w3, strides=[1,2,2,1], padding='SAME')
            x = bn(x, 512, is_training, is_training, name='bn3')
            x = tf.nn.leaky_relu(x, self.leaky)
            x = tf.reshape(x, [-1,16*512])
            mu = tf.matmul(x, w21) + b21
            logvar = tf.matmul(x, w22) + b22
            return mu, logvar

    def reparamenterize(self, mu, logvar, is_training):
        return tf.random_normal(tf.shape(mu), mean=mu, stddev=tf.exp(0.5*logvar)) if is_training else mu

    def decode(self, z, is_training, name='decoder'):
        if name in self.reuse.keys():
            reuse = self.reuse[name]
        else:
            self.reuse[name] = True
            reuse = False

        with tf.variable_scope(name, reuse=reuse) as scope:
            wfc = tf.get_variable(name='w_fc', shape=[self.latent_dim,16*512], dtype=tf.float32)
            bfc = tf.get_variable(name='b_fc', shape=[16*512], dtype=tf.float32)
            w1 = tf.get_variable(name='w_conv1', shape=[4,4,256,512], dtype=tf.float32)
            w2 = tf.get_variable(name='w_conv2', shape=[4,4,128,256], dtype=tf.float32)
            w3 = tf.get_variable(name='w_conv3', shape=[4,4,3,128], dtype=tf.float32)

            x = tf.matmul(z, wfc) + bfc
            x = tf.nn.leaky_relu(x, self.leaky)
            x = tf.reshape(x, [-1,4,4,512])
            bs = tf.shape(z)[0]
            x = tf.nn.conv2d_transpose(x, w1, output_shape=[bs,8,8,256], strides=[1,2,2,1], padding='SAME')
            x = bn(x, 256, is_training, is_training, name='bn1')
            x = tf.nn.leaky_relu(x, self.leaky)
            x = tf.nn.conv2d_transpose(x, w2, output_shape=[bs,16,16,128], strides=[1,2,2,1], padding='SAME')
            x = bn(x, 128, is_training, is_training, name='bn2')
            x = tf.nn.leaky_relu(x, self.leaky)
            x = tf.nn.conv2d_transpose(x, w3, output_shape=[bs,32,32,3], strides=[1,2,2,1], padding='SAME')
            x = bn(x, 3, is_training, is_training, name='bn3')
            x = tf.nn.sigmoid(x)
            return x

    def BCE(self, recon_x, x, mu, logvar):
        return -tf.reduce_mean(tf.reduce_sum(x * tf.log(recon_x+1e-8) + (1-x) * tf.log(1-recon_x+1e-8), axis=[1,2,3]))

    def KLD(self, mu, logvar):
        return tf.reduce_mean(tf.reduce_sum(-0.5 * (1 + logvar - tf.square(mu) - tf.exp(logvar)), axis=1))

    def build_graph(self, x, lr, kld_coef=1):
        self.mu, self.logvar = self.encode(x, True)
        self.zx = self.reparamenterize(self.mu, self.logvar, True)
        self.xzx = self.decode(self.zx, True)

        self.xz = self.decode(tf.random_normal([32,self.latent_dim]), False)

        self.bce_loss = self.BCE(self.xzx, x, self.mu, self.logvar)
        self.kld_loss = self.KLD(self.mu, self.logvar)
        self.total_loss = self.bce_loss + kld_coef * self.kld_loss
        self.mse_loss = tf.reduce_mean(tf.reduce_sum((x-self.xzx)**2, axis=[1,2,3]))

        self.train_step = tf.train.AdamOptimizer(lr).minimize(self.total_loss)
        return True

class Net(object):
    def __init__(self):
        super(Net, self).__init__()
        self.reuse = {}
        self.leaky = 0.1

    def classifier(self, name, x, keep_prob=1.0, is_training=False, update_batch_stats=False):
        if name in self.reuse.keys():
            reuse = self.reuse[name]
        else:
            self.reuse[name] = True
            reuse = False

        with tf.variable_scope(name, reuse=reuse) as scope:
            initializer = tf.contrib.layers.variance_scaling_initializer()
            w1 = tf.get_variable(name='w_conv1', shape=[3,3,3,128], dtype=tf.float32, initializer=initializer)
            b1 = tf.get_variable(name='b_conv1', shape=[128], dtype=tf.float32, initializer=tf.zeros_initializer)
            w2 = tf.get_variable(name='w_conv2', shape=[3,3,128,128], dtype=tf.float32, initializer=initializer)
            b2 = tf.get_variable(name='b_conv2', shape=[128], dtype=tf.float32, initializer=tf.zeros_initializer)
            w3 = tf.get_variable(name='w_conv3', shape=[3,3,128,128], dtype=tf.float32, initializer=initializer)
            b3 = tf.get_variable(name='b_conv3', shape=[128], dtype=tf.float32, initializer=tf.zeros_initializer)

            w4 = tf.get_variable(name='w_conv4', shape=[3,3,128,256], dtype=tf.float32, initializer=initializer)
            b4 = tf.get_variable(name='b_conv4', shape=[256], dtype=tf.float32, initializer=tf.zeros_initializer)
            w5 = tf.get_variable(name='w_conv5', shape=[3,3,256,256], dtype=tf.float32, initializer=initializer)
            b5 = tf.get_variable(name='b_conv5', shape=[256], dtype=tf.float32, initializer=tf.zeros_initializer)
            w6 = tf.get_variable(name='w_conv6', shape=[3,3,256,256], dtype=tf.float32, initializer=initializer)
            b6 = tf.get_variable(name='b_conv6', shape=[256], dtype=tf.float32, initializer=tf.zeros_initializer)

            w7 = tf.get_variable(name='w_conv7', shape=[3,3,256,512], dtype=tf.float32, initializer=initializer)
            b7 = tf.get_variable(name='b_conv7', shape=[512], dtype=tf.float32, initializer=tf.zeros_initializer)
            w8 = tf.get_variable(name='w_conv8', shape=[1,1,512,256], dtype=tf.float32, initializer=initializer)
            b8 = tf.get_variable(name='b_conv8', shape=[256], dtype=tf.float32, initializer=tf.zeros_initializer)
            w9 = tf.get_variable(name='w_conv9', shape=[1,1,256,128], dtype=tf.float32, initializer=initializer)
            b9 = tf.get_variable(name='b_conv9', shape=[128], dtype=tf.float32, initializer=tf.zeros_initializer)

            w10 = tf.get_variable(name='w_fc1', shape=[128,10], dtype=tf.float32, initializer=initializer)
            b10 = tf.get_variable(name='b_fc1', shape=[10], dtype=tf.float32, initializer=tf.constant_initializer(0.0))

            x = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding='SAME') + b1
            x = bn(x, 128, is_training, update_batch_stats, name='bn1')
            x = tf.nn.leaky_relu(x, self.leaky)
            x = tf.nn.conv2d(x, w2, strides=[1,1,1,1], padding='SAME') + b2
            x = bn(x, 128, is_training, update_batch_stats, name='bn2')
            x = tf.nn.leaky_relu(x, self.leaky)
            x = tf.nn.conv2d(x, w3, strides=[1,1,1,1], padding='SAME') + b3
            x = bn(x, 128, is_training, update_batch_stats, name='bn3')
            x = tf.nn.leaky_relu(x, self.leaky)
            x = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
            x = tf.nn.dropout(x, keep_prob)

            x = tf.nn.conv2d(x, w4, strides=[1,1,1,1], padding='SAME') + b4
            x = bn(x, 256, is_training, update_batch_stats, name='bn4')
            x = tf.nn.leaky_relu(x, self.leaky)
            x = tf.nn.conv2d(x, w5, strides=[1,1,1,1], padding='SAME') + b5
            x = bn(x, 256, is_training, update_batch_stats, name='bn5')
            x = tf.nn.leaky_relu(x, self.leaky)
            x = tf.nn.conv2d(x, w6, strides=[1,1,1,1], padding='SAME') + b6
            x = bn(x, 256, is_training, update_batch_stats, name='bn6')
            x = tf.nn.leaky_relu(x, self.leaky)
            x = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
            x = tf.nn.dropout(x, keep_prob)

            x = tf.nn.conv2d(x, w7, strides=[1,1,1,1], padding='SAME') + b7
            x = bn(x, 512, is_training, update_batch_stats, name='bn7')
            x = tf.nn.leaky_relu(x, self.leaky)
            x = tf.nn.conv2d(x, w8, strides=[1,1,1,1], padding='SAME') + b8
            x = bn(x, 256, is_training, update_batch_stats, name='bn8')
            x = tf.nn.leaky_relu(x, self.leaky)
            x = tf.nn.conv2d(x, w9, strides=[1,1,1,1], padding='SAME') + b9
            x = bn(x, 128, is_training, update_batch_stats, name='bn9')
            x = tf.nn.leaky_relu(x, self.leaky)
            x = tf.reduce_mean(x, [1,2])

            x = tf.matmul(x, w10) + b10
            x = tf.nn.softmax(x)
            return x

    def crossentropy(self, label, logits):
        return -tf.reduce_mean(tf.reduce_sum(label*tf.log(logits+1e-8), axis=1))

    def kldivergence(self, label, logits):
        return tf.reduce_mean(tf.reduce_sum(label*(tf.log(label+1e-8)-tf.log(logits+1e-8)), axis=1))

class NetSmall(object):
    def __init__(self):
        super(NetSmall, self).__init__()
        self.reuse = {}
        self.leaky = 0.1

    def classifier(self, name, x, keep_prob=1.0, is_training=False, update_batch_stats=False):
        if name in self.reuse.keys():
            reuse = self.reuse[name]
        else:
            self.reuse[name] = True
            reuse = False

        with tf.variable_scope(name, reuse=reuse) as scope:
            initializer = tf.contrib.layers.variance_scaling_initializer()
            w1 = tf.get_variable(name='w_conv1', shape=[3,3,3,96], dtype=tf.float32, initializer=initializer)
            b1 = tf.get_variable(name='b_conv1', shape=[96], dtype=tf.float32, initializer=tf.zeros_initializer)
            w2 = tf.get_variable(name='w_conv2', shape=[3,3,96,96], dtype=tf.float32, initializer=initializer)
            b2 = tf.get_variable(name='b_conv2', shape=[96], dtype=tf.float32, initializer=tf.zeros_initializer)
            w3 = tf.get_variable(name='w_conv3', shape=[3,3,96,96], dtype=tf.float32, initializer=initializer)
            b3 = tf.get_variable(name='b_conv3', shape=[96], dtype=tf.float32, initializer=tf.zeros_initializer)

            w4 = tf.get_variable(name='w_conv4', shape=[3,3,96,192], dtype=tf.float32, initializer=initializer)
            b4 = tf.get_variable(name='b_conv4', shape=[192], dtype=tf.float32, initializer=tf.zeros_initializer)
            w5 = tf.get_variable(name='w_conv5', shape=[3,3,192,192], dtype=tf.float32, initializer=initializer)
            b5 = tf.get_variable(name='b_conv5', shape=[192], dtype=tf.float32, initializer=tf.zeros_initializer)
            w6 = tf.get_variable(name='w_conv6', shape=[3,3,192,192], dtype=tf.float32, initializer=initializer)
            b6 = tf.get_variable(name='b_conv6', shape=[192], dtype=tf.float32, initializer=tf.zeros_initializer)

            w7 = tf.get_variable(name='w_conv7', shape=[3,3,192,192], dtype=tf.float32, initializer=initializer)
            b7 = tf.get_variable(name='b_conv7', shape=[192], dtype=tf.float32, initializer=tf.zeros_initializer)
            w8 = tf.get_variable(name='w_conv8', shape=[1,1,192,192], dtype=tf.float32, initializer=initializer)
            b8 = tf.get_variable(name='b_conv8', shape=[192], dtype=tf.float32, initializer=tf.zeros_initializer)
            w9 = tf.get_variable(name='w_conv9', shape=[1,1,192,192], dtype=tf.float32, initializer=initializer)
            b9 = tf.get_variable(name='b_conv9', shape=[192], dtype=tf.float32, initializer=tf.zeros_initializer)

            w10 = tf.get_variable(name='w_fc1', shape=[192,10], dtype=tf.float32, initializer=initializer)
            b10 = tf.get_variable(name='b_fc1', shape=[10], dtype=tf.float32, initializer=tf.constant_initializer(0.0))

            x = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding='SAME') + b1
            x = bn(x, 96, is_training, update_batch_stats, name='bn1')
            x = tf.nn.leaky_relu(x, self.leaky)
            x = tf.nn.conv2d(x, w2, strides=[1,1,1,1], padding='SAME') + b2
            x = bn(x, 96, is_training, update_batch_stats, name='bn2')
            x = tf.nn.leaky_relu(x, self.leaky)
            x = tf.nn.conv2d(x, w3, strides=[1,1,1,1], padding='SAME') + b3
            x = bn(x, 96, is_training, update_batch_stats, name='bn3')
            x = tf.nn.leaky_relu(x, self.leaky)
            x = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
            x = tf.nn.dropout(x, keep_prob)

            x = tf.nn.conv2d(x, w4, strides=[1,1,1,1], padding='SAME') + b4
            x = bn(x, 192, is_training, update_batch_stats, name='bn4')
            x = tf.nn.leaky_relu(x, self.leaky)
            x = tf.nn.conv2d(x, w5, strides=[1,1,1,1], padding='SAME') + b5
            x = bn(x, 192, is_training, update_batch_stats, name='bn5')
            x = tf.nn.leaky_relu(x, self.leaky)
            x = tf.nn.conv2d(x, w6, strides=[1,1,1,1], padding='SAME') + b6
            x = bn(x, 192, is_training, update_batch_stats, name='bn6')
            x = tf.nn.leaky_relu(x, self.leaky)
            x = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
            x = tf.nn.dropout(x, keep_prob)

            x = tf.nn.conv2d(x, w7, strides=[1,1,1,1], padding='SAME') + b7
            x = bn(x, 192, is_training, update_batch_stats, name='bn7')
            x = tf.nn.leaky_relu(x, self.leaky)
            x = tf.nn.conv2d(x, w8, strides=[1,1,1,1], padding='SAME') + b8
            x = bn(x, 192, is_training, update_batch_stats, name='bn8')
            x = tf.nn.leaky_relu(x, self.leaky)
            x = tf.nn.conv2d(x, w9, strides=[1,1,1,1], padding='SAME') + b9
            x = bn(x, 192, is_training, update_batch_stats, name='bn9')
            x = tf.nn.leaky_relu(x, self.leaky)
            x = tf.reduce_mean(x, [1,2])

            x = tf.matmul(x, w10) + b10
            x = tf.nn.softmax(x)
            return x

    def crossentropy(self, label, logits):
        return -tf.reduce_mean(tf.reduce_sum(label*tf.log(logits+1e-8), axis=1))

    def kldivergence(self, label, logits):
        return tf.reduce_mean(tf.reduce_sum(label*(tf.log(label+1e-8)-tf.log(logits+1e-8)), axis=1))
if __name__ == '__main__':
    main()
