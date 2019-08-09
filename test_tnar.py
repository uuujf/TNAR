import os
import math
import argparse
import numpy as np
import tensorflow as tf
from model import Net

parser = argparse.ArgumentParser()
parser.add_argument('--resume', type=str, default='./logs/vae_aug/model-120')
parser.add_argument('--datadir', type=str, default='/home/wjf/datasets/CIFAR10/SSL/seed123/')

args = parser.parse_args()
print(args)

# dataset
test_set = np.load(args.datadir+'/test.npz')
X_test = test_set['image'].reshape(-1,32,32,3);
Y_test = (np.eye(10)[test_set['label']]).astype(np.float32)

# tensor graph
net = Net()
x_test = tf.placeholder(tf.float32, [None,32,32,3], name='x_test')
y_test = tf.placeholder(tf.float32, [None,10], name='y_test')
out_test = net.classifier('net', x_test, keep_prob=1.0, is_training=False, update_batch_stats=False)
correct_prediction = tf.equal(tf.argmax(y_test,1), tf.argmax(out_test,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

weight_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='net')
saver = tf.train.Saver(var_list=weight_list)

gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    # sess.run(tf.global_variables_initializer())
    saver.restore(sess, args.resume); print('restored', args.resume)
    acc = 0
    for j in range(20):
        acc += sess.run(accuracy, feed_dict={x_test:X_test[500*j:500*(j+1)], y_test:Y_test[500*j:500*(j+1)]})
    acc /= 20
    print('test accuracy', acc)
