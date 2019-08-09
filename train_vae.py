import os
import math
import argparse
import numpy as np
import tensorflow as tf
from model import VAE

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--augment', action='store_true', default=False)
parser.add_argument('--iter-per-epoch', type=int, default=400)
parser.add_argument('--epoch', type=int, default=121)
parser.add_argument('--lr-decay-epoch', type=int, default=81)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--latent-dim', type=int, default=512)
parser.add_argument('--kld-coef', type=float, default=0.5)
parser.add_argument('--datadir', type=str, default='/home/wjf/datasets/CIFAR10/SSL/seed123/')
parser.add_argument('--logdir', type=str, default='./logs/vae_aug')

args = parser.parse_args()
print(args)
if not os.path.exists(args.logdir):
    os.makedirs(args.logdir)

# set random seed
tf.set_random_seed(args.seed)
np.random.seed(args.seed)

# dataset
unlabeled_train_set = np.load(args.datadir+'/unlabeled_train.npz')
X_train = unlabeled_train_set['image'].reshape(-1,32,32,3);

# build graph
lr = tf.placeholder(tf.float32, [], name='lr')
x_raw = tf.placeholder(tf.float32, [args.batch_size,32,32,3], name='x')
if args.augment:
    x = []
    for i in range(args.batch_size):
        xi = tf.pad(x_raw[i,:,:,:], [[2,2],[2,2],[0,0]])
        xi = tf.random_crop(xi, [32,32,3])
        xi = tf.image.random_flip_left_right(xi)
        x.append(xi)
    x = tf.stack(x, axis=0)
else:
    x = x_raw

vae = VAE(args.latent_dim)
vae.build_graph(x, lr, args.kld_coef)

saver = tf.train.Saver(max_to_keep=100)

gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(tf.global_variables_initializer(), feed_dict={lr:args.lr})
    for ep in range(args.epoch):
        if ep < args.lr_decay_epoch:
            decayed_lr = args.lr
        else:
            decayed_lr = args.lr * (args.epoch-ep)/float(args.epoch-args.lr_decay_epoch)

        for i in range(args.iter_per_epoch):
            mask = np.random.choice(len(X_train), args.batch_size, False)
            _, bce, kld = sess.run([vae.train_step, vae.bce_loss, vae.kld_loss], feed_dict={x_raw:X_train[mask], lr:decayed_lr})

        print('epoch:', ep, 'loss_bce:', bce, 'loss_kld:', kld)
        if ep % 10 == 0:
            saver.save(sess, os.path.join(args.logdir, 'model'), ep)
