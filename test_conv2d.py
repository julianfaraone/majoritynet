import numpy as np
import tensorflow as tf
import unittest
import sys
import argparse
import math

from tensorpack import *
from tensorpack.utils.argtools import shape2d, shape4d, get_data_format

from utils import Conv2D_mask, Conv2D_matmul, Conv2D_muladd 

# make sure file is in top directory to source models.givens_mm


HEIGHT=32
WIDTH=32
CHANNELS=16
FILTERS=8
K=3


def eval_on_data(model, data):

	pred_config = PredictConfig(
		model = model,
		input_names=["input"],
		output_names=["output"]
	)
	predictor = OfflinePredictor(pred_config)
	result = predictor(data)
	return result[0]

class Model(ModelDesc):

	def __init__(self, W, mode=0):
		self.mode = mode
		self.weight_init = tf.constant_initializer(W, dtype=tf.float32)

	input_dtype = tf.float32

	def inputs(self):
		return [tf.placeholder(self.input_dtype, [None, HEIGHT, WIDTH, CHANNELS], 'input')]
	""" 
	Need to implement build_graph
	"""
	def build_graph(self, image):
		if self.mode == 0:
			l = Conv2D('conv', image, FILTERS, K, kernel_initializer=self.weight_init, use_bias=False)
		elif self.mode == 1:
			l = Conv2D_matmul('conv', image, FILTERS, K, kernel_initializer=self.weight_init, use_bias=False)
		elif self.mode == 2:
			l = Conv2D_muladd('conv', image, FILTERS, K, kernel_initializer=self.weight_init, use_bias=False)

		logits = tf.identity(l, name='output')
		return logits	


class Conv2DTest(unittest.TestCase):

    def test(self):

        # input
        np.random.seed(23)
        data = np.random.randn(1, HEIGHT, WIDTH, CHANNELS)
        W = np.random.randn(K, K, CHANNELS, FILTERS)

        # Conv2D
        modelA = Model(W, mode=0)
        outA = eval_on_data(modelA, data)
        
        # Conv2D_matmul
        modelB = Model(W, mode=1)
        outB = eval_on_data(modelB, data)

        # Conv2D_muladd
        modelC = Model(W, mode=2)
        outC = eval_on_data(modelC, data)

        # print some data to verify equivalence
        print outA.shape
        print outA[0][0][0][:]
        print outB[0][0][0][:]
        print outC[0][0][0][:]

        # evaluate results
        success_a = np.allclose(outA, outB, atol=1e-4)
        success_b = np.allclose(outA, outC, atol=1e-4)
        self.assertEqual((success_a and success_b), True)



if __name__ == "__main__":

	#unittest.main()
	suite = Conv2DTest("test")
	unittest.TextTestRunner(verbosity=2).run( suite )




