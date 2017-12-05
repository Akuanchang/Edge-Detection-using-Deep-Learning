import os
# import sys
# sys.path.append('B:\Softs\deeplearningcodes\seismic-data-process\HolisticallyNestedEdgeDetection\MyHedEdgeDetect')
import time
# import wget
import numpy as np
from PIL import Image
from ioimage import  IO


class DataParser():

    def __init__(self, cfgs):

        self.io = IO()
        self.cfgs = cfgs
        self.train_file = os.path.join(cfgs['download_path'], cfgs['training']['list'])
        self.train_data_dir = os.path.join(cfgs['download_path'], cfgs['training']['dir'])
        self.training_pairs = self.io.read_file_list(self.train_file)

        self.samples = self.io.split_pair_names(self.training_pairs, self.train_data_dir)
        self.io.print_info('Training data set-up from {}'.format(os.path.join(self.train_file)))
        self.n_samples = len(self.training_pairs)

        self.all_ids =list(range(self.n_samples))#转换range 到list

        np.random.shuffle(self.all_ids)

        self.training_ids = self.all_ids[:int(self.cfgs['train_split'] * len(self.training_pairs))]
        self.num_training_ids = len(self.training_ids)

        self.validation_ids = self.all_ids[int(self.cfgs['train_split'] * len(self.training_pairs)):]
        self.num_validation_ids = len(self.validation_ids)

        self.io.print_info('Training samples {}'.format(len(self.training_ids)))
        self.io.print_info('Validation samples {}'.format(len(self.validation_ids)))

    def get_training_batch(self):

        batch_ids = np.random.choice(self.training_ids, self.cfgs['batch_size_train'])

        return self.get_batch(batch_ids)

    # 顺序选择文件名列表中的一个batch

    def next_training_batch(self, indexs):

        # dimx = np.shape(self.training_ids)

        batch_ids = self.training_ids[indexs*self.cfgs['batch_size_train']:self.cfgs['batch_size_train']*(indexs+1)]

        return self.get_batch(batch_ids)




    def get_validation_batch(self):

        batch_ids = np.random.choice(self.validation_ids, self.cfgs['batch_size_val'])

        return self.get_batch(batch_ids)

    def get_batch(self, batch):

        tstart = time.time()

        filenames = []
        images = []
        edgemaps = []


        for idx, b in enumerate(batch):

            im = Image.open(self.samples[b][0])
            em = Image.open(self.samples[b][1])

            im = im.resize((self.cfgs['training']['image_width'], self.cfgs['training']['image_height']))
            em = em.resize((self.cfgs['training']['image_width'], self.cfgs['training']['image_height']))

            im = np.array(im, dtype=np.float32)
            im = im[:, :, self.cfgs['channel_swap']]
            im -= self.cfgs['mean_pixel_value']# cdk change here for train test

            # Labels needs to be 1 or 0 (edge pixel or not)
            # or can use regression targets as done by the author
            # https://github.com/s9xie/hed/blob/9e74dd710773d8d8a469ad905c76f4a7fa08f945/src/caffe/layers/image_labelmap_data_layer.cpp#L213

            em = np.array(em.convert('L'), dtype=np.float32)



            if self.cfgs['target_regression']:
                bin_em = em / 255.0


            else:
                bin_em = np.zeros_like(em)
                bin_em[np.where(em)] = 1

            # Some edge maps have 3 channels some don't
            bin_em = bin_em if bin_em.ndim == 2 else bin_em[:, :, 0]
            # To fit [batch_size, H, W, 1] output of the network
            bin_em = np.expand_dims(bin_em, 2)

            images.append(im)
            edgemaps.append(bin_em)
            filenames.append(self.samples[b])

        return images, edgemaps, filenames

