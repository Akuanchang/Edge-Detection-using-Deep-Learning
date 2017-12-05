#coding=UTF-8
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import os
from deconv import deconv_mobile_version
from cross_entropy import  class_balanced_sigmoid_cross_entropy
import time
from data import DataParser
from ioimage import IO
import yaml

from PIL import Image
import urllib
import io

# 定义模型
class Vggnet():

	def __init__(self,cfgs):
		self.cfgs =cfgs
		self.io = IO()

	# 定义网络
	def hed_net(self,image_input):
		with tf.variable_scope('SDFU-net',  [image_input]):
			with slim.arg_scope([slim.conv2d],
			                    activation_fn=tf.nn.relu,
			                    weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
			                    weights_regularizer=slim.l2_regularizer(0.0005)):
				# 卷积和池化层 vgg16 conv && max_pool layers
				im_shape = tf.shape(image_input)
				start_t = time.time()
				with tf.name_scope(name='conv1'):
					net_out = slim.repeat(image_input, 2, slim.conv2d, 4, [3, 3], scope='conv1')
					dsn1 = net_out
					net_out = slim.max_pool2d(net_out, [2, 2], scope='pool1')

				with tf.name_scope(name='conv2'):
					net_out = slim.repeat(net_out, 2, slim.conv2d, 4, [3, 3], scope='conv2')
					dsn2 = net_out
					net_out = slim.max_pool2d(net_out, [2, 2], scope='pool2')
				with tf.name_scope(name='conv3'):
					net_out = slim.repeat(net_out, 3, slim.conv2d, 4, [3, 3], scope='conv3')
					dsn3 = net_out
					net_out = slim.max_pool2d(net_out, [2, 2], scope='pool3')
				#
				with tf.name_scope(name='conv4'):
					net_out = slim.repeat(net_out, 3, slim.conv2d, 4, [3, 3], scope='conv4')
					dsn4 = net_out
					net_out = slim.max_pool2d(net_out, [2, 2], scope='pool4')
				#
				with tf.name_scope(name='conv5'):
					net_out = slim.repeat(net_out, 3, slim.conv2d, 4, [3, 3], scope='conv5')
					dsn5 = net_out
				   # net = slim.max_pool2d(net, [2,2], scope='pool5') #不需要这一池化层

				# dsn layers(deep supervision nets)
				dsn1 = slim.conv2d(dsn1, 1, [1, 1], scope='dsn1',activation_fn= None  )

				# dsn1不需要deconv

				dsn2 = slim.conv2d(dsn2, 1, [1, 1], scope='dsn2',activation_fn= None)

				# deconv_shape = tf.pack([batch_size, image_height, image_width,1])

				dsn2 = deconv_mobile_version(dsn2, 2, [im_shape[0],im_shape[1], im_shape[2],1])

				dsn3 = slim.conv2d(dsn3, 1, [1, 1], scope='dsn3',activation_fn= None)
				# deconv_shape = tf.pack([batch_size, image_height, image_width, 1])

				dsn3 = deconv_mobile_version(dsn3, 4, [im_shape[0],im_shape[1], im_shape[2],1])
				#
				dsn4 = slim.conv2d(dsn4, 1, [1, 1], scope='dsn4',activation_fn= None)
				# deconv_shape = tf.pack([batch_size, image_height, image_width, 1])
				dsn4 = deconv_mobile_version(dsn4, 8, [im_shape[0],im_shape[1], im_shape[2],1])
				#
				dsn5 = slim.conv2d(dsn5, 1, [1, 1], scope='dsn5',activation_fn= None)
				# deconv_shape = tf.pack([batch_size, image_height, image_width, 1])
				dsn5 = deconv_mobile_version(dsn5, 16, [im_shape[0],im_shape[1], im_shape[2],1])

				# dsn fuse
				# dsn_fuse = tf.concat(3, [dsn1, dsn2, dsn3, dsn4, dsn5])

				dsn_fuse = tf.concat([dsn1,dsn2,dsn3,dsn4,dsn5],3)

				sideoutput = [dsn1,dsn2,dsn3,dsn4,dsn5]

				dsn_fuse = tf.reshape(dsn_fuse, [im_shape[0],im_shape[1], im_shape[2], 5])  ##without this, will get error: ValueError: Number of in_channels must be known.

				dsn_fuse = slim.conv2d(dsn_fuse, 1, [1, 1], scope='dsn_fuse',activation_fn= None)
		# 		complete output maps from sidelayer and fuse layer

				self.io.print_info('Build model finished : {:.4f}s'.format(time.time()-start_t))
		# return dsn_fuse, dsn1, dsn2, dsn3, dsn4, dsn5

		return sideoutput,dsn_fuse

	def setup_testing(self, session,sideoutput,dsn_fuse):

		"""
			Apply sigmoid non-linearity to side layer ouputs + fuse layer outputs for predictions
		"""

		self.predictions = []
		self.outputs = sideoutput + [dsn_fuse]

		for idx, b in enumerate(self.outputs):
			output = tf.nn.sigmoid(b, name='output_{}'.format(idx))

			self.predictions.append(output)

	def setup_training(self,session,sideoutput,dsn_fuse,labels):
		self.predictions = []
		self.loss = 0

		self.io.print_warning('Deep supervision application set to {}'.format(self.cfgs['deep_supervision']))
		self.io.print_warning(
			'Targets are {}'.format('continous' if self.cfgs['deep_supervision'] else 'binary {0, 1}'))
		with tf.name_scope(name='loss'):
			for idx ,b in enumerate(sideoutput):
				output = tf.nn.sigmoid(b,name='output_{}'.format(idx))

				# cost = class_balanced_sigmoid_cross_entropy(output,labels,name="cross_entropy_{}".format(idx))
				cost = (1/self.cfgs['batch_size_train'])*tf.nn.l2_loss(output-labels,name="L2_loss_layers")
				self.predictions.append(output)

				if self.cfgs['deep_supervision']:
					self.loss += (self.cfgs['loss_weights']*cost)
			fuse_output = tf.nn.sigmoid(dsn_fuse,name='fuse')

			# fuse_cost = class_balanced_sigmoid_cross_entropy(fuse_output,labels,name='cross_entropy_fuse')
			fuse_cost =(1/self.cfgs['batch_size_train'])*tf.nn.l2_loss(fuse_output - labels,name='L2_loss_fuse_layer')

			self.predictions.append(fuse_output)
			self.loss += (self.cfgs['loss_weights']*fuse_cost)

			# pred = tf.cast(tf.greater(fuse_output, 0.5), tf.int32, name='predictions')
			# error = tf.cast(tf.not_equal(pred, tf.cast(labels, tf.int32)), tf.float32)
			# self.error = tf.reduce_mean(error, name='pixel_error')
			self.error = (1/self.cfgs['batch_size_val'])*tf.nn.l2_loss((fuse_output -labels), name='validation_error')

		tf.summary.scalar('loss', self.loss)
		tf.summary.scalar('error', self.error)
		self.merged_summary = tf.summary.merge_all()

		self.train_writer = tf.summary.FileWriter(self.cfgs['save_dir'] + '/train', session.graph)
		self.val_writer = tf.summary.FileWriter(self.cfgs['save_dir'] + '/val')












