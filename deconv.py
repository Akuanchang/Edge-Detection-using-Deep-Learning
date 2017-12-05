import tensorflow as tf
import numpy as np


def get_kernel_size(factor):
	"""
	Find the kernel size given the desired factor of upsampling.
	"""
	return 2 * factor - factor % 2


def upsample_filt(size):
	"""
	Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
	"""
	factor = (size + 1) // 2
	if size % 2 == 1:
		center = factor - 1
	else:
		center = factor - 0.5
	og = np.ogrid[:size, :size]
	return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)


def bilinear_upsample_weights(factor, number_of_classes):
	"""
	Create weights matrix for transposed convolution with bilinear filter
	initialization.
	"""
	filter_size = get_kernel_size(factor)

	weights = np.zeros((filter_size,
	                    filter_size,
	                    number_of_classes,
	                    number_of_classes), dtype=np.float32)

	upsample_kernel = upsample_filt(filter_size)

	for i in range(number_of_classes):
		weights[:, :, i, i] = upsample_kernel

	return weights


def deconv(inputs, upsample_factor):
    input_shape = tf.shape(inputs)

    # Calculate the ouput size of the upsampled tensor
    upsampled_shape = tf.pack([input_shape[0],
                               input_shape[1] * upsample_factor,
                               input_shape[2] * upsample_factor,
                               1])

    upsample_filter_np = bilinear_upsample_weights(upsample_factor, 1)
    upsample_filter_tensor = tf.constant(upsample_filter_np)

    # Perform the upsampling
    upsampled_inputs = tf.nn.conv2d_transpose(inputs, upsample_filter_tensor,
                                              output_shape=upsampled_shape,
                                              strides=[1, upsample_factor, upsample_factor, 1])

    return upsampled_inputs

def deconv_mobile_version(inputs, upsample_factor, upsampled_shape):

    upsample_filter_np = bilinear_upsample_weights(upsample_factor, 1)
    upsample_filter_tensor = tf.constant(upsample_filter_np)

    # Perform the upsampling
    upsampled_inputs = tf.nn.conv2d_transpose(inputs, upsample_filter_tensor,
                                              output_shape=upsampled_shape,
                                              strides=[1, upsample_factor, upsample_factor, 1])

    return upsampled_inputs