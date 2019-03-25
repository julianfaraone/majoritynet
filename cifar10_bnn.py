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

from utils import get_dorefa, Conv2D_matmul, Conv2D_muladd, bnn

"""
DoreFa-Net ResNet-20 for Cifar10, should get close to 8.75% err (0.27M params)
https://arxiv.org/abs/1512.03385
"""

BATCH_SIZE = 50 #1024 #50
BITW = 1
BITA = 1
BITG = 32


class Model(ModelDesc):
    def inputs(self):
        return [tf.placeholder(tf.float32, [None, 32, 32, 3], 'input'),
                tf.placeholder(tf.int32, [None], 'label')]

    def build_graph(self, image, label):
        image = image / 128.0

        """
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
        """

        def get_logits(x):
            x = bnn.layer('l1', x, 128, filter_size=[3, 3])
            x = bnn.layer('l2', x, 128, filter_size=[3, 3], pool=([2, 2], [2, 2]))
            x = bnn.layer('l3', x, 256, filter_size=[3, 3])
            x = bnn.layer('l4', x, 256, filter_size=[3, 3], pool=([2, 2], [2, 2]))
            x = bnn.layer('l5', x, 512, filter_size=[3, 3])
            x = bnn.layer('l6', x, 512, filter_size=[3, 3], pool=([2, 2], [2, 2]))

            # fully connected
            x = bnn.layer('l7', x, 1024, filter_size=[4, 4], padding='VALID')
            x = bnn.layer('l8', x, 1024)
            x = bnn.layer('l9', x, 10, activate='none')
            _y = tf.identity(x)
            y = tf.reshape(_y, [-1, 10])
            return y



        #with remap_variables(new_get_variable), \
        with argscope(BatchNorm, decay=0.9, epsilon=1e-4), \
                argscope(Conv2D, use_bias=False, nl=tf.identity, 
                    kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out')):

            logits = get_logits(image)


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

    """
    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=args.learning_rate[0], trainable=False)
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        add_moving_summary(lr)
        return opt
    """

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=args.learning_rate[0], trainable=False)
        """
        lr = tf.train.exponential_decay(
            learning_rate=lr,
            global_step=tf.constant(1, dtype=tf.float32),
            decay_steps=500,
            decay_rate=dr, staircase=True, name='learning_rate')
        """
        add_moving_summary(lr)
        return tf.train.AdamOptimizer(lr)


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
        output_names=['wrong_vector', 'l1/bX', 'l1/bW', 'l2/bX', 'l2/bW', 'l2/xW']
    )
    pred = SimpleDatasetPredictor(pred_config, ds)
    acc1 = RatioCounter()
    for o in pred.get_result():
        batch_size = o[0].shape[0]
        acc1.feed(o[0].sum(), batch_size)
        a1 = o[1]
        wq1 = o[2]
        a2 = o[3]
        wq2 = o[4]
        w2 = o[5]
    print("Top1 Error: {}".format(acc1.ratio))
    
    # quantise w2 and compare with graph quantisation
    E = np.sum(np.mean(np.abs(w2))) 
    wQ = np.where(np.equal(w2, 0), np.ones_like(w2), np.sign(w2/E))*E

    print wq2.shape, wQ.shape
    print np.allclose(wQ[0][0][0], wq2[0][0][0], rtol=1e-2)


def dump(model, sessinit, ds):

    # get model weights
    weight_list = ['l1/bW',
                    'l2/bW',
                    'l3/bW',
                    'l4/bW',
                    'l5/bW',
                    'l6/bW',
                    'l7/bW',
                    'l8/bW',
                    'l9/bW']

    # get model activations
    act_list = ['input']

    # get model batch norm params
    bn_gamma_list = ['l1/BatchNorm/gamma:0',
                    'l2/BatchNorm/gamma:0',
                    'l3/BatchNorm/gamma:0',
                    'l4/BatchNorm/gamma:0',
                    'l5/BatchNorm/gamma:0',
                    'l6/BatchNorm/gamma:0',
                    'l7/BatchNorm/gamma:0',
                    'l8/BatchNorm/gamma:0',
                    'l9/BatchNorm/gamma:0']

    bn_beta_list = ['l1/BatchNorm/beta:0',
                    'l2/BatchNorm/beta:0',
                    'l3/BatchNorm/beta:0',
                    'l4/BatchNorm/beta:0',
                    'l5/BatchNorm/beta:0',
                    'l6/BatchNorm/beta:0',
                    'l7/BatchNorm/beta:0',
                    'l8/BatchNorm/beta:0',
                    'l9/BatchNorm/beta:0']

    def weight_quant_np(w):
        E = np.sum(np.mean(np.abs(w)))
        wQ = np.where(np.equal(w, 0), np.ones_like(w), np.sign(w/E))*E
        return wQ


    def get_node(act, w, bn_g, bn_b):

        pred_config = PredictConfig(
            model=model,
            session_init=sessinit,
            input_names=['input', 'label'],
            output_names=[act, w, bn_g, bn_b, 'label']
        )
        pred = SimpleDatasetPredictor(pred_config, ds)
        acc1 = RatioCounter()

        av = []
        lb = []
        for i, o in enumerate(pred.get_result()):
            av.append(o[0])
            wt = o[1]
            bt_g = o[2]
            bt_b = o[3]
            lb.append(o[4])
            if (i*BATCH_SIZE >= 1000): #limits number of dumped acts
                break

        data_dict = np.vstack(av), np.hstack(lb), {w: weight_quant_np(wt), bn_g: bt_g, bn_b: bt_b}
        return data_dict

    fd_name = 'cifar10_bnn-w{}-a{}-g{}'.format(BITW,BITA,BITG)
    os.system('mkdir -p dump_log/{}'.format(fd_name))

    for i in range(9): #18
        print "fetching: ", act_list[0], weight_list[i], bn_gamma_list[i], bn_beta_list[i]
        img, lbl, out = get_node(act_list[0], weight_list[i], bn_gamma_list[i], bn_beta_list[i])
        fname = 'dump_log/{}/node{}.pkl'.format(fd_name, i+1)
        with open(fname, 'wb') as fh:
            pickle.dump(out, fh)
    # dump the image
    fname = 'dump_log/{}/input.pkl'.format(fd_name)
    with open(fname, 'wb') as fh:
        pickle.dump(img, fh)

    fname = 'dump_log/{}/lbl.pkl'.format(fd_name)
    with open(fname, 'wb') as fh:
        pickle.dump(lbl, fh)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model for training')
    parser.add_argument('--eval', action='store_true', help='run offline evaluation instead of training')
    parser.add_argument('--dump', action='store_true', help='dump weights and activations')
    parser.add_argument('--dorefa',
                        help='number of bits for W,A,G, separated by comma. Defaults to \'1,1,32\'',
                        default='1,1,32')
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
        save_dir = os.path.join('train_log', 'cifar10_bnn-w{}-a{}-g{}'.format(BITW,BITA,BITG))
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
                #ScheduledHyperParamSetter('learning_rate', 
                #    zip(args.num_epochs[:-1], args.learning_rate[1:]))
                                          #[(1, 0.1), (82, 0.01), (123, 0.001), (300, 0.0002)])
                HyperParamSetterWithFunc( 'learning_rate', lambda e, x: x*(3e-7/x)**(1.0/args.num_epochs[-1]) )

            ],
            max_epoch=args.num_epochs[-1],
            session_init=SaverRestore(args.load) if args.load else None
        )
        num_gpu = max(get_num_gpu(), 1)
        launch_train_with_config(config, SyncMultiGPUTrainerParameterServer(num_gpu))
