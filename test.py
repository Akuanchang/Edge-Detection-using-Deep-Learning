import os
import io
import sys

import yaml
# import urlparse
import urllib
# import StringIO
# import cStringIO
import numpy as np
from PIL import Image
import tensorflow as tf
from  vgg_model import Vggnet
from ioimage import IO


# 测试模型
class test():

	def __init__(self,config_file):
		self.io = IO()
		self.init = True

		try:
			pfile = open(config_file)
			self.cfgs = yaml.load(pfile)
			pfile.close()
		except Exception as err:
			self.io.print_error('Error reading config file {}，{}'.format(config_file),err)

	def setup(self,session):
		try:
			self.model = Vggnet(self.cfgs)
			self.io.print_info('Done initializing Vggnet model')

		except Exception as err:
			self.io.print_error('Error setting up Vggnet model, {}'.format(err))
			self.init = False

	def run(self,session,run = "testing"):

		if not self.init:
			return

		self.images = tf.placeholder(tf.float32, [None, self.cfgs[run]['image_height'], self.cfgs[run]['image_width'],
		                                          self.cfgs[run]['n_channels']])
		self.edgemaps = tf.placeholder(tf.float32,
		                               [None, self.cfgs[run]['image_height'], self.cfgs[run]['image_width'], 1])

		sideoutput, dsn_fuse = self.model.hed_net(self.images)

		self.model.setup_testing(session,sideoutput,dsn_fuse)

		session.run(tf.global_variables_initializer())

		# 重载模型
		meta_model_file = os.path.join(self.cfgs['save_dir'],
									   'models/hed-model-{}'.format(self.cfgs['test_snapshot']))
		saver = tf.train.Saver()
		saver.restore(session, meta_model_file)
		self.io.print_info('Done restoring Vggnet model from {}'.format(meta_model_file))


		filepath = os.path.join(self.cfgs['download_path'], self.cfgs['testing']['list'])
		test_list = self.io.read_file_list(filepath)

		self.io.print_info('Writing PNGs at {}'.format(self.cfgs['test_output']))

		for idx,img in enumerate(test_list):

			test_filename = os.path.join(self.cfgs['download_path'], self.cfgs['testing']['dir'], img)
			print(test_filename)
			im = self.capture_pixels(test_filename)
			print('shape1', np.shape(im))

			edgemap = session.run(self.model.predictions, feed_dict={self.images: [im]})

			print(np.shape(edgemap))

			self.save_egdemaps(edgemap, idx)

			self.io.print_info('Done testing {}, {}'.format(test_filename, im.shape))

	def save_egdemaps(self, em_maps, index):

		# Take the edge map from the network from side layers and fuse layer
		print(np.shape(em_maps))
		em_maps = [e[0] for e in em_maps]
		print(np.shape(em_maps))
		em_maps = em_maps + [np.mean(np.array(em_maps), axis=0)]
		print("cdk", np.shape(em_maps))

		for idx, em in enumerate(em_maps):
			em[em < self.cfgs['testing_threshold']] = 0.0

			em = 255.0 * (1.0 - em)
			em = np.tile(em, [1, 1, 3])
			print("cdk2", np.shape(em_maps))

			em = Image.fromarray(np.uint8(em))
			print("cdk3", np.shape(em_maps))
			em.save(os.path.join(self.cfgs['test_output'], 'testing-{}-{:03}.png'.format(index, idx)))

	def fetch_image(self, test_image):

		# is url
		image = None

		if not urllib.urlparse.urlparse(test_image).scheme == "":

			url_response = urllib.urlopen(test_image)

			if url_response.code == 404:
				print(
					self.io.print_error('[Testing] URL error code : {1} for {0}'.format(test_image, url_response.code)))
				return None

			try:

				image_buffer = io.StringIO(url_response.read())
				image = self.capture_pixels(image_buffer)

			except Exception as err:

				print(self.io.print_error('[Testing] Error with URL {0} {1}'.format(test_image, err)))
				return None

		# read from disk
		elif os.path.exists(test_image):

			try:

				fid = open(test_image, 'r')
				stream = fid.read()
				fid.close()

				image_buffer = io.StringIO(stream)
				image = self.capture_pixels(image_buffer)

			except Exception as err:

				print(self.io.print_error('[Testing] Error with image file {0} {1}'.format(test_image, err)))
				return None

		return image

	def capture_pixels(self, image_buffer):
		if os.path.exists(image_buffer):
			try:
				image = Image.open(image_buffer)
				image = image.resize((self.cfgs['testing']['image_width'], self.cfgs['testing']['image_height']))
				image = np.array(image, np.float32)
				image = self.colorize(image)

				image = image[:, :, self.cfgs['channel_swap']]
				image -= self.cfgs['mean_pixel_value']
			except Exception as err:
				print(self.io.print_error('[Testing] Error with image file {0} {1}'.format(image_buffer, err)))
				return None

		return image

	def colorize(self, image):

		# BW to 3 channel RGB image
		if image.ndim == 2:
			image = image[:, :, np.newaxis]
			image = np.tile(image, (1, 1, 3))
		elif image.shape[2] == 4:
			image = image[:, :, :3]

		return image