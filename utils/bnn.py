import tensorflow as tf
import numpy as np

# code

# binary activation function
#def activation(x):
#    x = tf.clip_by_value(x, -1.0, 1.0)
#    return x + tf.stop_gradient(tf.sign(x) - x)

def activation(x):
    x = tf.clip_by_value(x, -1.0, 1.0)
    noise_percentage = 0.1
    zeros = 0*x
    zeros = tf.reduce_sum(zeros, 0)
    ones = zeros + 1
    noise = tf.where(tf.random_uniform(zeros.shape) < noise_percentage, -1*ones, ones)
    noisy_x = x*noise
    return x + tf.stop_gradient(tf.sign(noisy_x) - x)

# create weight + bias variables with update op as in BinaryNet
def weight_bias(shape, binary=True):
    print(shape)
    init = np.random.uniform(-1, 1, size=shape) #tf.random_uniform(shape, -1.0, 1.0)
    #x = tf.Variable(init)
    x = tf.get_variable('xW', shape, initializer=tf.constant_initializer(init))
    b = tf.constant(0.1, shape=[shape[-1]])

    if binary:
        y = tf.get_variable('yW', shape, initializer=tf.constant_initializer(init)) # floating point copy

        coeff = np.float32(1./np.sqrt(1.5/ (np.prod(shape[:-2]) * (shape[-2] + shape[-1]))))
        print(coeff)

        tmp = y + coeff * (x - y)
        tmp = tf.clip_by_value(tmp, -1.0, 1.0)
        tmp = tf.group(x.assign(tmp), y.assign(tmp))
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, tmp)

        x = tf.clip_by_value(x, -1.0, 1.0)
        xbin = tf.sign(x) * tf.reduce_mean(tf.abs(x)) #, axis=[0, 1, 2]) # maybe uncomment this for training
        x = x + tf.stop_gradient(xbin - x)

    return x, b #tf.Variable(tf.constant(0.1, shape=[shape[-1]]))

def batch_norm(x, epsilon, decay=0.9):
    return tf.contrib.layers.batch_norm(x, decay=decay, center=True, scale=True,
        epsilon=epsilon, updates_collections=None, trainable=True,
        fused=True)

# a layer in BinaryNet
def layer(name, x, num_output, filter_size=[1, 1], stride=[1, 1], pool=None, activate='bin',
          binary=True, norm=True, epsilon=0.0001, padding='SAME'):
    shape = filter_size + [x.shape[-1].value, num_output]

    with tf.variable_scope(name):
        W, b = weight_bias(shape, binary)

        Wi = tf.identity(W, name='bW')
        xi = tf.identity(x, name='bX')


        x = tf.nn.conv2d(x, W, strides=[1, stride[0], stride[1], 1], padding=padding) + b

        if activate == 'bin':
            if pool is not None:
                x = tf.nn.max_pool(x, ksize=[1, pool[0][0], pool[0][1], 1], strides=[1, pool[-1][0], pool[-1][1], 1], padding='VALID')

            if norm:
                x = batch_norm(x, epsilon)
        else:
            if norm:
                x = batch_norm(x, epsilon)

            if pool is not None:
                x = tf.nn.max_pool(x, ksize=[1, pool[0][0], pool[0][1], 1], strides=[1, pool[-1][0], pool[-1][1], 1], padding='VALID')

    if activate == 'bin':
        return activation(x)
    elif activate == 'relu':
        return tf.nn.relu(x)

    assert(activate == 'none')
    return x
