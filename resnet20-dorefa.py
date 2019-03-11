#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: resnet-dorefa.py

import argparse
import numpy as np
import os
import cv2
import tensorflow as tf
import pickle

from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary, add_activation_summary
from tensorpack.utils.gpu import get_num_gpu
from tensorpack.dataflow import dataset
from tensorpack.tfutils import argscope, get_model_loader, model_utils
from tensorpack.utils.stats import RatioCounter
from tensorpack.tfutils.varreplace import remap_variables

from utils import get_dorefa, Conv2D_matmul, Conv2D_muladd

"""
DoreFa-Net ResNet-20 for Cifar10, should get close to 8.75% err (0.27M params)
https://arxiv.org/abs/1512.03385
"""
BATCH_SIZE = 1024

BITW = 1
BITA = 1
BITG = 32


class Model(ModelDesc):
    def inputs(self):
        return [tf.placeholder(tf.float32, [None, 32, 32, 3], 'input'),
                tf.placeholder(tf.int32, [None], 'label')]

    def build_graph(self, image, label):
        image = image / 128.0

        fw, fa, fg = get_dorefa(BITW, BITA, BITG)

        def new_get_variable(v):
            name = v.op.name
            # don't binarize first and last layer
            if not name.endswith('W') or 'conv1' in name or 'fct' in name:
                return v
            else:
                logger.info("Binarizing weight {}".format(v.op.name))
                return fw(v)

        def nonlin(x):
            return tf.clip_by_value(x, 0.0, 1.0)

        def activate(x, name='act'):
            return tf.identity(fa(nonlin(x)), name=name)

        def resblock(x, channel, stride):
            def get_stem_full(x):
                l = Conv2D('c3x3a', x, channel, 3)
                l = BatchNorm('stembn', l)
                l = activate(l, name='act_b')
                l = Conv2D('c3x3b', l, channel, 3)
                #add_activation_summary(l, ['histogram'], name=l.op.name)
                #tf.add_to_collection('ACTS', l)
                return l

            channel_mismatch = channel != x.get_shape().as_list()[3]
            if stride != 1 or channel_mismatch or 'pool1' in x.name:
                # handling pool1 is to work around an architecture bug in our model
                if stride != 1 or 'pool1' in x.name:
                    x = AvgPooling('pool', x, stride, stride)
                x = BatchNorm('bn', x)
                x = activate(x, name='act_a')
                shortcut = Conv2D('shortcut', x, channel, 1)
                stem = get_stem_full(x)
            else:
                shortcut = x
                x = BatchNorm('bn', x)
                x = activate(x, name='act_a')
                stem = get_stem_full(x)
            return shortcut + stem

        def group(name, x, channel, nr_block, stride):
            with tf.variable_scope(name + 'blk1'):
                x = resblock(x, channel, stride)
            for i in range(2, nr_block + 1):
                with tf.variable_scope(name + 'blk{}'.format(i)):
                    x = resblock(x, channel, 1)
            return x

        with remap_variables(new_get_variable), \
                argscope(BatchNorm, decay=0.9, epsilon=1e-4), \
                argscope(Conv2D, use_bias=False, nl=tf.identity, 
                    kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out')):

            #acts = tf.get_collection('ACTS')

            l = Conv2D('conv1', image, 16, 3, stride=1, padding='VALID', use_bias=True)
            l = group('conv2', l, 16, 3, 1)
            l = group('conv3', l, 32, 3, 2)
            l = group('conv4', l, 64, 3, 2)
            l = BatchNorm('lastbn', l)
            l = nonlin(l)
            l = GlobalAvgPooling('gap', l)
            logits = FullyConnected('fct', l, 10, 
                kernel_initializer=tf.random_normal_initializer(stddev=0.01))


        tf.nn.softmax(logits, name='output')

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        wrong = tf.cast(tf.logical_not(tf.nn.in_top_k(logits, label, 1)), tf.float32, name='wrong_vector')
        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        add_moving_summary(cost)

        #for a in acts:
        #    print a.op.name

        add_param_summary(('.*/W', ['histogram']))   # monitor W
        return tf.identity(cost, name='cost')


    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=args.learning_rate[0], trainable=False)
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        add_moving_summary(lr)
        return opt


def get_data(train_or_test):
    isTrain = train_or_test == 'train'
    ds = dataset.Cifar10(train_or_test)
    pp_mean = ds.get_per_pixel_mean()
    if isTrain:
        augmentors = [
            imgaug.CenterPaste((40, 40)),
            imgaug.RandomCrop((32, 32)),
            imgaug.Flip(horiz=True),
            imgaug.MapImage(lambda x: x - pp_mean),
        ]
    else:
        augmentors = [
            imgaug.MapImage(lambda x: x - pp_mean)
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
    if isTrain:
        ds = PrefetchData(ds, 3, 2)
    return ds


def eval_on_cifar10(model, sessinit, ds):
    pred_config = PredictConfig(
        model=model,
        session_init=sessinit,
        input_names=['input', 'label'],
        output_names=['wrong_vector', 'conv2blk2/c3x3b/output:0', 'conv2blk2/b_input',
        'conv2blk2/a_bnorm', 'conv2blk2/c3x3a/output:0', 'conv2blk2/c3x3b/W:0']
    )
    pred = SimpleDatasetPredictor(pred_config, ds)
    acc1 = RatioCounter()
    for o in pred.get_result():
        batch_size = o[0].shape[0]
        acc1.feed(o[0].sum(), batch_size)
        actA = o[1]
        actB = o[2]
        actC = o[3]
        actD = o[4]
        wA = o[5]
    print("Top1 Error: {}".format(acc1.ratio))
    print actA[0][0][0]
    print actB[0][0][0]
    print actC[0][0][0]
    print actD[0][0][0]


    x = wA
    E = np.sum(np.mean(np.abs(x)))
    wQ = np.where(np.equal(x, 0), np.ones_like(x), np.sign(x/E))*E

    print actB[0].shape
    print wQ.shape, x.shape
    print actA[0].shape

    
    with tf.Session():
        img = tf.constant(actB[0].reshape(1, 30, 30, 16), dtype=tf.float32)
        kern = tf.constant(wQ.astype(np.float32), dtype=tf.float32)
        out = tf.nn.conv2d(img, kern, [1,1,1,1], padding='SAME')
        ao = out.eval()

    print ao[0][0][0]
    print np.sum(np.abs(ao[0]-actA[0]))
    print np.allclose(ao[0], actA[0], rtol=1e-2)


def dump(model, sessinit, ds):

    # get model weights
    weight_list = ['conv2blk1/c3x3a/W:0', 'conv2blk1/c3x3b/W:0',
                    'conv2blk2/c3x3a/W:0', 'conv2blk2/c3x3b/W:0',
                    'conv2blk3/c3x3a/W:0', 'conv2blk3/c3x3b/W:0',
                    'conv3blk1/c3x3a/W:0', 'conv3blk1/c3x3b/W:0',
                    'conv3blk2/c3x3a/W:0', 'conv3blk2/c3x3b/W:0',
                    'conv3blk3/c3x3a/W:0', 'conv3blk3/c3x3b/W:0',
                    'conv4blk1/c3x3a/W:0', 'conv4blk1/c3x3b/W:0',
                    'conv4blk2/c3x3a/W:0', 'conv4blk2/c3x3b/W:0',
                    'conv4blk3/c3x3a/W:0', 'conv4blk3/c3x3b/W:0']

    # get model activations
    act_list = ['conv2blk1/act_a', 'conv2blk1/act_b',
                    'conv2blk2/act_a', 'conv2blk2/act_b',
                    'conv2blk3/act_a', 'conv2blk3/act_b',
                    'conv3blk1/act_a', 'conv3blk1/act_b',
                    'conv3blk2/act_a', 'conv3blk2/act_b',
                    'conv3blk3/act_a', 'conv3blk3/act_b',
                    'conv4blk1/act_a', 'conv4blk1/act_b',
                    'conv4blk2/act_a', 'conv4blk2/act_b',
                    'conv4blk3/act_a', 'conv4blk3/act_b']

    def weight_quant_np(w):
        x = w
        E = np.sum(np.mean(np.abs(x)))
        wQ = np.where(np.equal(x, 0), np.ones_like(x), np.sign(x/E))*E
        return wQ


    def get_node(act, w):

        pred_config = PredictConfig(
            model=model,
            session_init=sessinit,
            input_names=['input', 'label'],
            output_names=[act, w]
        )
        pred = SimpleDatasetPredictor(pred_config, ds)
        acc1 = RatioCounter()

        av = []
        for i, o in enumerate(pred.get_result()):
            av.append(o[0])
            wt = o[1]
            if (i*BATCH_SIZE >= 1000): #limits number of dumped acts
                break

        data_dict = {act: np.vstack(av), w: weight_quant_np(wt)}
        return data_dict

    fd_name = 'resnet20-w{}-a{}-g{}'.format(BITW,BITA,BITG)
    os.system('mkdir -p dump_log/{}'.format(fd_name))

    for i in range(18):
        print "fetching: ", act_list[i], weight_list[i]
        out = get_node(act_list[i], weight_list[i])
        fname = 'dump_log/{}/node{}.pkl'.format(fd_name, i)
        with open(fname, 'wb') as fh:
            pickle.dump(out, fh)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model for training')
    parser.add_argument('--eval', action='store_true', help='run offline evaluation instead of training')
    parser.add_argument('--dump', action='store_true', help='dump weights and activations')
    parser.add_argument('--dorefa',
                        help='number of bits for W,A,G, separated by comma. Defaults to \'1,4,32\'',
                        default='1,4,32')
    parser.add_argument('--num_epochs', nargs="+", type=int, default=[150,250,350], help='epoch learning rate schedule')
    parser.add_argument('--learning_rate', nargs="+", type=float, default=[1e-1,1e-2,1e-3], help='learning rate schedule')
    parser.add_argument('--label', help='append string for log_dir', type=str, default='NA')

    args = parser.parse_args()
    
    BITW, BITA, BITG = map(int, args.dorefa.split(','))

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    #logger.auto_set_dir()

    model = Model()
    dataset_train = get_data('train')
    dataset_test = get_data('test')

    if args.eval:
        eval_on_cifar10(model, get_model_loader(args.load), dataset_test)

    elif args.dump:
        dump(model, get_model_loader(args.load), dataset_test)

    else:
        save_dir = os.path.join('train_log', 'resnet20-w{}-a{}-g{}'.format(BITW,BITA,BITG))
        if args.label:
            save_dir = save_dir + "-{}".format(args.label)
        logger.set_logger_dir(save_dir)

        config = TrainConfig(
            model=model,
            dataflow=dataset_train,
            callbacks=[
                ModelSaver(max_to_keep=2),
                MinSaver('validation_error'),
                InferenceRunner(dataset_test,
                                [ScalarStats('cost'), ClassificationError('wrong_vector')]),
                ScheduledHyperParamSetter('learning_rate', 
                    zip(args.num_epochs[:-1], args.learning_rate[1:]))
                                          #[(1, 0.1), (82, 0.01), (123, 0.001), (300, 0.0002)])
            ],
            max_epoch=args.num_epochs[-1],
            session_init=SaverRestore(args.load) if args.load else None
        )
        num_gpu = max(get_num_gpu(), 1)
        launch_train_with_config(config, SyncMultiGPUTrainerParameterServer(num_gpu))
