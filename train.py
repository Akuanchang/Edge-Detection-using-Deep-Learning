
import os
import sys
import yaml
import argparse
import tensorflow as tf
# from termcolor import colored
import numpy as np
import time
from ioimage import IO
from vgg_model import Vggnet
from data import DataParser


# 训练模型
class train():
	def __init__(self,config_file):

		self.io = IO()
		self.init = True

		try:
			pfile = open(config_file)
			self.cfgs = yaml.load(pfile)
			pfile.close()
		except Exception as err:

			self.io.print_error('Error reading config file {}, {}'.format(config_file,err))



	def setup(self):
		try:
			self.model = Vggnet(self.cfgs)
			self.io.print_info('Done initializing Vggnet model')

			dirs = ['train','val','test','models']
			dirs = [os.path.join(self.cfgs['save_dir'] + '/{}'.format(d)) for d in dirs]
			_ = [os.makedirs(d) for d in dirs if not os.path.exists(d)]

		except Exception as err:
			self.io.print_error('Error setting up Vggnet model, {}'.format(err))
			self.init = False

	def run(self,session,run='training'):

		if not self.init:
			return

		self.images = tf.placeholder(tf.float32, [None, self.cfgs[run]['image_height'], self.cfgs[run]['image_width'],
		                                          self.cfgs[run]['n_channels']])
		self.edgemaps = tf.placeholder(tf.float32,
		                               [None, self.cfgs[run]['image_height'], self.cfgs[run]['image_width'], 1])

		train_data = DataParser(self.cfgs)


		total_batches = train_data.num_training_ids//self.cfgs['batch_size_train']

		sideoutput, dsn_fuse = self.model.hed_net(self.images)

		self.model.setup_training(session,sideoutput,dsn_fuse,self.edgemaps)

		d_step = 100 * total_batches
		global_step = tf.Variable(0)
		init_learn_rate = self.cfgs['optimizer_params']['learning_rate']
		learn_rate = tf.train.exponential_decay(init_learn_rate, global_step=global_step,
												decay_steps=d_step, decay_rate=0.75, staircase=True)

		train = tf.train.AdamOptimizer(learn_rate).minimize(self.model.loss, global_step=global_step)

		# opt = tf.train.AdamOptimizer(self.cfgs['optimizer_params']['learning_rate'])
        #
		# train = opt.minimize(self.model.loss)

		loss_log = []
		total_loss_log = []
		error_log = []
		ep = []
		batchs = []
		ep_v = []

		session.run(tf.global_variables_initializer())

		timestart = time.clock()


		print(' beign training')


		for idx in range(self.cfgs['max_iterations']):
			avg_loss = 0
			ep.append(idx)

			for b in range(total_batches):
				im,em,_ = train_data.next_training_batch(b)
				run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
				run_metadata = tf.RunMetadata()

				_,summary,loss = session.run([train,self.model.merged_summary,self.model.loss],
											 feed_dict = {self.images : im,self.edgemaps:em},
											 options=run_options,
											 run_metadata=run_metadata
											 )
				if (idx + 1) * (b + 1) % d_step == 0:
					lr = session.run(learn_rate)
					self.io.print_info('Learning_rate:{}'.format(lr))

				loss_log.append(loss)
				batchs.append(idx * total_batches + b)

				avg_loss += loss
				# if b+1 % 5 == 0:
					# self.io.print_info('[{}/{}] TRAINING loss : {}'.format(idx + 1, self.cfgs['max_iterations'], loss))
				self.io.print_info('[{}/{}] TRAINING loss : {}'.format(b + 1, total_batches, loss))


			total_loss = avg_loss / total_batches
			self.io.print_info(
				'[{}/{}] TRAINING average_loss : {}'.format(idx + 1, self.cfgs['max_iterations'], total_loss))

			total_loss_log.append(total_loss)

			self.model.train_writer.add_run_metadata(run_metadata, 'step{:06}'.format(idx))
			self.model.train_writer.add_summary(summary, idx)

			if idx+1 % self.cfgs['save_interval'] == 0:
				saver = tf.train.Saver()
				saver.save(session,os.path.join(self.cfgs['save_dir'],'models/hed-model'),global_step=idx)

			if idx+1 % self.cfgs['val_interval'] == 0:
				im,em ,_ = train_data.get_validation_batch()

				summary,error = session.run([self.model.merged_summary,self.model.error],
											feed_dict = {self.images:im,self.edgemaps:em})
				self.model.val_writer.add_summary(summary,idx)
				self.io.print_info('[{}/{}] VALIDATION error : {}'.format(idx + 1, self.cfgs['max_iterations'], error))
				ep_v.append(idx)
				error_log.append(error)
		endtime = time.clock()
		self.io.print_info('Train cost time : %f hours' % ((endtime - timestart) / 3600))
		np.savetxt(os.path.join(self.cfgs['cost_log_file'], self.cfgs['cost_log']['loss']), [batchs, loss_log])
		np.savetxt(os.path.join(self.cfgs['cost_log_file'], self.cfgs['cost_log']['avgloss']), [ep, total_loss_log])
		np.savetxt(os.path.join(self.cfgs['cost_log_file'], self.cfgs['cost_log']['error']), [ep_v, error_log])

		self.model.train_writer.close()