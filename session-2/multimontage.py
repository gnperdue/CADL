# Session 3 - Training a Network w/ TF
import sys
if sys.version_info < (3, 4):
    print('This version of Python is too old. Please update to at least 3.4.')

# numpy/scipy libs
try:
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.transform import resize
except ImportError:
    print('You are missing some libraies!')
    
# tf
try:
    import tensorflow as tf
except ImportError:
    print('You need TensorFlow!')
    
try:
    from libs import utils, gif
except ImportError:
    print('Make sure you have the libs stuff!')


# Next we get `linear` from the `utils` we imported... also need to get
# `flatten` from utils (we'll write them out)
def flatten(x, name=None, reuse=None):
    with tf.variable_scope('flatten'):
        dims = x.get_shape().as_list()
        if len(dims) == 4:
            flattened = tf.reshape(x, shape=[-1, dims[1] * dims[2] * dims[3]])
        elif len(dims) == 2 or len(dims) == 1:
            flattened = x
        else:
            raise ValueError('Expected n dimensions of 1, 2, or 4. '
                             'Found:', len(dims))
    return flattened


def linear(x, n_output, name=None, activation=None, reuse=None):
    """
    scope is called 'name', number of inputs is derived from the shape of `x`
    """
    if len(x.get_shape()) != 2:
        x = flatten(x, reuse=reuse)
    n_input = x.get_shape().as_list()[1]
    with tf.variable_scope(name or "fc", reuse=reuse):
        W = tf.get_variable(name='W',
                            shape=[n_input, n_output],
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name='b',
                            shape=[n_output],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(0.0))
        h = tf.nn.bias_add(name='h',
                           value=tf.matmul(x, W),
                           bias=b)
        if activation:
            h = activation(h)
        return h, W


def split_image(img):
    """
    img needs 2d shape
    """
    xs = []  # positions
    ys = []  # colors
    for row_i in range(img.shape[0]):
        for col_i in range(img.shape[1]):
            xs.append([row_i, col_i])
            ys.append(img[row_i, col_i])
            
    xs = np.array(xs)
    ys = np.array(ys)
    return xs, ys


def build_model(xs, ys, n_neurons, n_layers,
                activation_fn, final_activation_fn, cost_type):
    """
    * xs, ys are _lists_ of coordinates and (here) pulse heights
    * n_layers does not count the final layer
    * we have n_neurons per created layer except for the final layer
    
    We use `linear` (and `flatten`) functions from above...
    """
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    if xs.ndim != 2:
        raise ValueError('xs should be n_obs x n_features, or 2d')
    if ys.ndim != 2:
        raise ValueError('ys should be n_obs x n_features, or 2d')
        
    n_xs = xs.shape[1]
    n_ys = ys.shape[1]
    
    X = tf.placeholder(name='X', shape=[None, n_xs], dtype=tf.float32)
    Y = tf.placeholder(name='Y', shape=[None, n_ys], dtype=tf.float32)
    
    current_input = X
    for layer_i in range(n_layers):
        # get back h, W, but don't keep W here...
        current_input = linear(current_input,
                               n_neurons,
                               activation=activation_fn,
                               name='layer{}'.format(layer_i))[0]
        
    Y_pred = linear(current_input,
                    n_ys,
                    activation=final_activation_fn,
                    name='pred')[0]
    
    if cost_type == 'l1_norm':
        cost = tf.reduce_mean(tf.reduce_sum(tf.abs(Y - Y_pred), 1))
    elif cost_type == 'l2_norm':
        cost = tf.reduce_mean(
            tf.reduce_sum(tf.squared_difference(Y, Y_pred), 1)
        )
    else:
        raise ValueError('Unknown cost type {}'.format(cost_type))
        
    return {
        'X': X, 'Y': Y, 'Y_pred': Y_pred, 'cost': cost
    }


def train(imgs,
          learning_rate=0.0001,
          batch_size=200,
          n_iterations=10,
          gif_step=2,
          n_neurons=30,
          n_layers=10,
          activation_fn=tf.nn.relu,
          final_activation_fn=tf.nn.tanh,
          cost_type='l2_norm'):
    N, H, W, C = imgs.shape
    print('all_xs, all_ys prep...')
    all_xs, all_ys = [], []
    for img_i, img in enumerate(imgs):
        xs, ys = split_image(img)
        all_xs.append(np.c_[xs, np.repeat(img_i, [xs.shape[0]])])
        all_ys.append(ys)
    print(' all_xs shape = {}'.format(np.shape(all_xs)))
    print(' all_ys shape = {}'.format(np.shape(all_ys)))

    # here I think we use `(-1, 3)` for the shape instead of `(-1, 2)`
    # because we are including an extra input for the image index.
    xs = np.array(all_xs).reshape(-1, 3)
    # don't think we need to normalize inputs when they're coordinates...
    # xs = (xs - np.mean(xs, 0)) / np.std(xs, 0)
    # pk has `(-1, 3)` here instead of `(-1, 1)` because he has rgb images
    ys = np.array(all_ys).reshape(-1, 1)
    # ys already normalized for my data
    print(' xs shape = {}'.format(np.shape(xs)))
    print(' ys shape = {}'.format(np.shape(ys)))
    
    g = tf.Graph()
    with tf.Session(graph=g) as sess:
        model = build_model(xs, ys, n_neurons, n_layers,
                            activation_fn, final_activation_fn, cost_type)
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate
        ).minimize(model['cost'])
        sess.run(tf.initialize_all_variables())
        gifs = []
        costs = []
        step_i = 0
        for it_i in range(n_iterations):
            idxs = np.random.permutation(range(len(xs)))
            n_batches = len(idxs) // batch_size
            training_cost = 0
            for batch_i in range(n_batches):
                idxs_i = idxs[batch_i * batch_size: (batch_i + 1) * batch_size]
                cost = sess.run([model['cost'], optimizer],
                                feed_dict={model['X']: xs[idxs_i],
                                           model['Y']: ys[idxs_i]})[0]
                training_cost += cost
                
            print('iteration {}/{}: cost {}'.format(
                it_i + 1, n_iterations, training_cost / n_batches
            ))
        
            # every gif_step iters, draw the prediction of our
            # input xs, which should try to recreate the image
            if (it_i + 1) % gif_step == 0:
                costs.append(training_cost / n_batches)
                ys_pred = model['Y_pred'].eval(
                    feed_dict={model['X']: xs}, session=sess
                )
                img = ys_pred.reshape(imgs.shape)
                gifs.append(img)

    gifs = np.array(gifs)
    return gifs


tf.reset_default_graph()

imgs = np.load('img_data.npy')
print('We are working with an images tensor with shape {}'.format(imgs.shape))

imgs_t = np.array(imgs).copy()
print("imgs_t.shape = {}".format(imgs_t.shape))
imgs_t = imgs_t.reshape([imgs_t.shape[0], imgs_t.shape[1], imgs_t.shape[2], 1])
print("imgs_t.shape = {} (after reshape)".format(imgs_t.shape))

gifs = train(imgs=imgs_t, n_iterations=4)
print("np.array(gifs).shape = {}".format(np.array(gifs).shape))

gifs = np.reshape(gifs,
                  (gifs.shape[0], gifs.shape[1], gifs.shape[2], gifs.shape[3]))
print("np.array(gifs).shape = {} (post reshape)".format(
    np.array(gifs).shape
))

# we're gonna need a montage!
montage_gifs = [np.clip(utils.montage(g), 0, 1) for g in gifs]
_ = gif.build_gif(montage_gifs, saveto='multiple.gif')

# notebook only
# ipyd.Image(url='multiple.gif?{}'.format(np.random.rand()),
#            height=500, width=500)

final = gifs[:, :, :, 49]
final_gif = [np.clip(f, 0, 1) for f in final]
_ = gif.build_gif(final_gif, saveto='final.gif')

# notebook only
# ipyd.Image(url='final.gif?{}'.format(np.random.rand()),
#            height=200, width=200)
