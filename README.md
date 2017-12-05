# Edge-Detection-using-Deep-Learning
Edge Detection using Deep Learning using tensorflow_gpu


Author = {'Chang, Dekuan'}
Email  = {"cdk2708@gmail.com"}


Input image                |  Final fused Edge maps    |    Edge maps from side layers
:-------------------------:|:-------------------------:|:-------------------------:

This repository contains tensorflow implementation of the [HED model](https://github.com/s9xie/hed). 

Details of hyper-paramters are available in the [paper](https://arxiv.org/pdf/1504.06375.pdf)

    @InProceedings{xie15hed,
      author = {"Xie, Saining and Tu, Zhuowen"},
      Title = {Holistically-Nested Edge Detection},
      Booktitle = "Proceedings of IEEE International Conference on Computer Vision",
      Year  = {2015},
    }

## Get this repo
```
git clone https://github.com/harsimrat-eyeem/holy-edge.git
```

## Installing requirements
Its recommended to install the requirements in a [conda virtual environment](https://conda.io/docs/using/envs.html#create-an-environment)

## Setting up

The HED model is trained on [augmented training](http://vcl.ucsd.edu/hed/HED-BSDS.tar) set created by the authors.
```
# location where training data : http://vcl.ucsd.edu/hed/HED-BSDS.tar would be downloaded and decompressed
download_path: '<path>'
# location of snapshot and tensorbaord summary events
save_dir: '<path>'
# location where to put the generated edgemaps during testing
test_output: '<path>'
```
## Training data & Models
You can train the model to simply generate edgemaps.

This downloads the augmented training set created by authors of HED. Augmentation strategies
 include rotation to 16 predefined angles and cropping largest rectangle from the image. Details in section (4.1). To download training data run

## VGG-16 base model
VGG base model available [here](https://github.com/machrisaa/tensorflow-vgg) is used for producing multi-level features.
 The model is modified according with Section (3.) of the [paper](https://arxiv.org/pdf/1504.06375.pdf). 
 Deconvolution layers are set with [tf.nn.conv2d_transpose](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d_transpose). T
 he model uses single deconvolution layer in each side layers.
 
## Training

Launch training
```
parser.add_argument('--train', dest='run_train', action='store_true', default=True, help='Launch training')
parser.add_argument('--test', dest='run_test', action='store_true', default=False, help='Launch testing on a list of images')
```
Launch tensorboard
```
tensorboard --logdir=<save_dir>
```

## Testing
Edit the snapshot you want to use for testing in `hed/configs/hed.yaml`

parser.add_argument('--train', dest='run_train', action='store_true', default=False, help='Launch training')
parser.add_argument('--test', dest='run_test', action='store_true', default=True, help='Launch testing on a list of images')

```
test_snapshot: <snapshot number>
```

save_dir: <path_to_repo_on_disk>/hed
test_snapshot: 50
# location where to put the generated edgemaps during testing
test_output: '<path>'
```


