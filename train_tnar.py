import os
import math
import argparse
import numpy as np
import tensorflow as tf
from model import VAE,Net

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--augment', action='store_true', default=False)
parser.add_argument('--iter-per-epoch', type=int, default=400)
parser.add_argument('--epoch', type=int, default=501)
parser.add_argument('--lr-decay-epoch', type=int, default=461)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--batch-size-ul', type=int, default=128)
parser.add_argument('--mom1', type=float, default=0.9)
parser.add_argument('--mom2', type=float, default=0.5)
parser.add_argument('--keep-prob', type=float, default=0.5)
parser.add_argument('--coef-vat1', type=float, default=1.0)
parser.add_argument('--coef-vat2', type=float, default=1.0)
parser.add_argument('--coef-ent', type=float, default=1.0)
parser.add_argument('--zeta', type=float, default=0.001)
parser.add_argument('--epsilon1', type=float, default=5.0)
parser.add_argument('--epsilon2', type=float, default=1.0)
parser.add_argument('--resume', type=str, default='./logs/vae/model-120')
parser.add_argument('--latent-dim', type=int, default=512)
parser.add_argument('--datadir', type=str, default='/home/wjf/datasets/CIFAR10/SSL/seed123/')
parser.add_argument('--logdir', type=str, default='./logs/tnar')

args = parser.parse_args()
print(args)
if not os.path.exists(args.logdir):
    os.makedirs(args.logdir)

# set random seed
tf.set_random_seed(args.seed)
np.random.seed(args.seed)

# dataset
labeled_train_set = np.load(args.datadir+'/labeled_train.npz')
unlabeled_train_set = np.load(args.datadir+'/unlabeled_train.npz')
valid_set = np.load(args.datadir+'/valid.npz')
X_train = labeled_train_set['image'].reshape(-1,32,32,3);
Y_train = (np.eye(10)[labeled_train_set['label']]).astype(np.float32)
Xul_train = unlabeled_train_set['image'].reshape(-1,32,32,3);
X_valid = valid_set['image'].reshape(-1,32,32,3);
Y_valid = (np.eye(10)[valid_set['label']]).astype(np.float32)

#================== build train graph
lr = tf.placeholder(tf.float32, [], name='lr')
momentum = tf.placeholder(tf.float32, [], name='mom')
y = tf.placeholder(tf.float32, [None,10], name='y')

x_raw = tf.placeholder(tf.float32, [args.batch_size,32,32,3], name='x')
x_ul_raw = tf.placeholder(tf.float32, [args.batch_size_ul,32,32,3], name='xul')
if args.augment:
    x = []
    for i in range(args.batch_size):
        xi = tf.pad(x_raw[i,:,:,:], [[2,2],[2,2],[0,0]])
        xi = tf.random_crop(xi, [32,32,3])
        xi = tf.image.random_flip_left_right(xi)
        x.append(xi)
    x = tf.stack(x, axis=0)
    x_ul = []
    for i in range(args.batch_size_ul):
        xi = tf.pad(x_ul_raw[i,:,:,:], [[2,2],[2,2],[0,0]])
        xi = tf.random_crop(xi, [32,32,3])
        xi = tf.image.random_flip_left_right(xi)
        x_ul.append(xi)
    x_ul = tf.stack(x_ul, axis=0)
else:
    x = x_raw
    x_ul = x_ul_raw

vae = VAE(args.latent_dim)
net = Net()
out = net.classifier('net', x, keep_prob=args.keep_prob, is_training=True, update_batch_stats=True)
out_ul = net.classifier('net', x_ul, keep_prob=args.keep_prob, is_training=True, update_batch_stats=False)
mu, logvar = vae.encode(x_ul, False)
z = vae.reparamenterize(mu, logvar, False)
x_recon = vae.decode(z, False)

r0 = tf.zeros_like(z, name='zero_holder')
x_recon_r0 = vae.decode(z+r0, False)
diff2 = 0.5 * tf.reduce_sum((x_recon - x_recon_r0)**2, axis=[1,2,3])
diffJaco = tf.gradients(diff2, r0)[0]
def normalizevector(r):
    shape = tf.shape(r)
    r = tf.reshape(r, [shape[0],-1])
    r /= (1e-12+tf.reduce_max(tf.abs(r), axis=1, keepdims=True))
    r / tf.sqrt(tf.reduce_sum(r**2, axis=1, keepdims=True)+1e-6)
    return tf.reshape(r, shape)

# power method
r_adv = normalizevector(tf.random_normal(shape=tf.shape(z)))
for j in range(1):
    r_adv = 1e-6*r_adv
    x_r = vae.decode(z+r_adv, False)
    out_r = net.classifier('net', x_r-x_recon+x_ul, keep_prob=args.keep_prob, is_training=True, update_batch_stats=False)
    kl = net.kldivergence(out_ul, out_r)
    r_adv = tf.stop_gradient(tf.gradients(kl, r_adv)[0]) / 1e-6
    r_adv = normalizevector(r_adv)
    # begin cg------->
    rk = r_adv + 0
    pk = rk + 0
    xk = tf.zeros_like(rk)
    for k in range(4):
        Bpk = tf.stop_gradient(tf.gradients(diffJaco*pk, r0)[0])
        pkBpk = tf.reduce_sum(pk*Bpk, axis=1, keepdims=True)
        rk2 = tf.reduce_sum(rk*rk, axis=1, keepdims=True)
        alphak = (rk2 / (pkBpk+1e-8)) * tf.cast((rk2>1e-8), tf.float32)
        xk += alphak * pk
        rk -= alphak * Bpk
        betak = tf.reduce_sum(rk*rk, axis=1, keepdims=True) / (rk2+1e-8)
        pk = rk + betak * pk
    # end cg<---------
    r_adv = normalizevector(xk)
x_adv = vae.decode(z+r_adv*args.epsilon1, False)
r_x = x_adv - x_recon
out_adv = net.classifier('net', x_ul+r_x, keep_prob=args.keep_prob, is_training=True, update_batch_stats=False)
r_x = normalizevector(r_x)

r_adv_orth = normalizevector(tf.random_normal(shape=tf.shape(x_ul)))
for j in range(1):
    r_adv_orth1 = 1e-6*r_adv_orth
    out_r = net.classifier('net', x_ul+r_adv_orth1, keep_prob=args.keep_prob, is_training=True, update_batch_stats=False)
    kl = net.kldivergence(out_ul, out_r)
    r_adv_orth1 = tf.stop_gradient(tf.gradients(kl, r_adv_orth1)[0]) / 1e-6
    r_adv_orth = r_adv_orth1 - args.zeta*(tf.reduce_sum(r_x*r_adv_orth,axis=[1,2,3],keepdims=True)*r_x) + args.zeta*r_adv_orth
    r_adv_orth = normalizevector(r_adv_orth)
out_adv_orth = net.classifier('net', x_ul+r_adv_orth*args.epsilon2, keep_prob=args.keep_prob, is_training=True, update_batch_stats=False)

vat_loss = net.kldivergence(tf.stop_gradient(out_ul), out_adv)
vat_loss_orth = net.kldivergence(tf.stop_gradient(out_ul), out_adv_orth)
en_loss = net.crossentropy(out_ul, out_ul)
ce_loss = net.crossentropy(y, out)
total_loss = ce_loss + args.coef_vat1*vat_loss + args.coef_vat2*vat_loss_orth + args.coef_ent*en_loss

weight_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='net')
train_step=tf.train.AdamOptimizer(lr, beta1=momentum).minimize(total_loss, var_list=weight_list)
#================== train graph

#================== build validate graph
x_val = tf.placeholder(tf.float32, [None,32,32,3], name='x_val')
y_val = tf.placeholder(tf.float32, [None,10], name='y_val')
out_val = net.classifier('net', x_val, keep_prob=1.0, is_training=False, update_batch_stats=False)
correct_prediction = tf.equal(tf.argmax(y_val,1), tf.argmax(out_val,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#================== validate graph

# restore vae weights
encoder_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder')
decoder_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder')
vae_saver = tf.train.Saver(var_list=encoder_list+decoder_list)
# vae weights
saver = tf.train.Saver(max_to_keep=100)

# optimize
gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(tf.global_variables_initializer(), feed_dict={lr:args.lr, momentum:args.mom1})
    vae_saver.restore(sess, args.resume); print('restored', args.resume)
    for ep in range(args.epoch):
        if ep < args.lr_decay_epoch:
            decayed_lr = args.lr
            decayed_mom = args.mom1
        else:
            decayed_lr = args.lr * (args.epoch-ep)/float(args.epoch-args.lr_decay_epoch)
            decayed_mom = args.mom2

        for i in range(args.iter_per_epoch):
            mask = np.random.choice(len(X_train), args.batch_size, False)
            mask_ul = np.random.choice(len(Xul_train), args.batch_size_ul, False)
            _ = sess.run(train_step, feed_dict={x_raw:X_train[mask], y:Y_train[mask], x_ul_raw:Xul_train[mask_ul], lr:decayed_lr, momentum:decayed_mom})

        acc_valid = 0
        for j in range(int(1000/500)):
            acc_valid += sess.run(accuracy, feed_dict={x_val:X_valid[500*j:500*(j+1)], y_val:Y_valid[500*j:500*(j+1)]})
        acc_valid /= 2

        print('epoch', ep, 'acc_valid', acc_valid)
        if ep % 50 == 0:
            saver.save(sess, os.path.join(args.logdir, 'model'), ep)
