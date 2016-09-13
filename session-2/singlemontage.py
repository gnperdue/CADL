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


# Next we get `linear` from the `utils` we imported...
# Let's write them out instead.
# also need to get `flatten` from utils
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


imgs = np.load('img_data.npy')
print('We are working with an images tensor with shape {}'.format(imgs.shape))

# Hmmm, well, actually, we're already normalized for `ys`.
xs, ys = split_image(imgs[49])
# We seem to need the reshape to `(N, 1)` for everything to work.
ys = np.reshape(ys, [len(ys), 1])


tf.reset_default_graph()
X = tf.placeholder(tf.float32, shape=[None, 2], name='X')
# pk has `[None, 3]` here for `Y`, for color output
Y = tf.placeholder(tf.float32, shape=[None, 1], name='Y')

# Now, let's create a 6-layer deep network with 20 neurons in each layer.
# ```
# h1 = phi(X W1 + b1)
# h2 = phi(h1 W2 + b2)
# ...
# h6 = phi(h5 W6 + b6)
# Yhat = phi(H6 W7 + b7)
# ```

# len==8, pk's ends with 3, of course
n_neurons = [2, 20, 20, 20, 20, 20, 20, 1]
current_input = X
for layer_i in range(1, len(n_neurons)):
    # loop starting after the data input layer, use `linear` function to build
    # params, etc.
    layer_name = 'layer_' + str(layer_i)
    act_fun = tf.nn.relu if (layer_i + 1) < len(n_neurons) else None
    # we also get back W, but we don't need it
    current_input, _ = linear(x=current_input,
                              n_output=n_neurons[layer_i],
                              name=layer_name,
                              activation=act_fun)
Yhat = current_input

assert(X.get_shape().as_list() == [None, 2])
assert(Yhat.get_shape().as_list() == [None, 1])
assert(Y.get_shape().as_list() == [None, 1])


# Cost Function
#   \begin{equation}
#   \text{cost}(Y, \hat{Y}) = \frac{1}{B} \sum_{b = 0}^B E_b
#   \end{equation}
# where the error is measured as:
#   \begin{equation}
#   E = \sum_{c=0}^C (Y_c - \hat{Y_c})^2
#   \end{equation}
# This is the $l_2$ (ell-two) loss.
error = tf.squared_difference(Y, Yhat)
assert(error.get_shape().as_list() == [None, 1])
sum_error = tf.reduce_sum(error, reduction_indices=1)
assert(sum_error.get_shape().as_list() == [None])
cost = tf.reduce_mean(sum_error)
assert(cost.get_shape().as_list() == [])


# Now we need an optimizer.
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)


# TF, do your thing...

n_iterations = 100
batch_size = 10
learned_imgs = []
costs = []
gif_step = n_iterations // 10
step_i = 0
n_batches = len(xs) // batch_size

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for it_i in range(n_iterations):
        idxs = np.random.permutation(range(len(xs)))
        for batch_i in range(n_batches):
            idxs_i = idxs[batch_i * batch_size: (batch_i + 1) * batch_size]
            training_cost = sess.run([cost, optimizer],
                                     feed_dict={X: xs[idxs_i],
                                                Y: ys[idxs_i]})[0]
            
        if (it_i + 1) % gif_step == 0:
            costs.append(training_cost / n_batches)
            ys_pred = Yhat.eval(feed_dict={X: xs}, session=sess)
            img = np.clip(ys_pred.reshape(imgs[49].shape), 0, 1)
            learned_imgs.append(img)
            # fig, ax = plt.subplots(1, 2)
            # ax[0].plot(costs)
            # ax[0].set_xlabel('Iteration')
            # ax[0].set_ylabel('Cost')
            # ax[1].imshow(img)
            # ax[1].axis('off')
            # fig.suptitle('Iteration {}'.format(it_i))
            # plt.show()

# save the bundle of imgs as a GIF
_ = gif.build_gif(learned_imgs, saveto='single.gif', show_gif=False)
