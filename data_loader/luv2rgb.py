# third-parties library imports
import tensorflow as tf
import tensorflow_probability as tfp
from enum import Enum
import numpy as np

from tensorflow.python.ops.image_ops_impl import ResizeMethod

class ChannelDepth(Enum):
    """Channel depth to float value."""

    C8_BIT = 255.0
    C12_BIT = 4095.0

def convert_luv_to_rgb(
    luma: tf.Tensor,
    chroma: tf.Tensor,
    luma_depth: ChannelDepth = ChannelDepth.C12_BIT,
    chroma_depth:ChannelDepth = ChannelDepth.C8_BIT,
    rgb_depth: ChannelDepth = ChannelDepth.C8_BIT
) -> tf.Tensor:
    """Convert a MPC3Evo LUV image into a RGB image.

    Args:
        luma (tf.Tensor): Luminance channel of the image.
        chroma (tf.Tensor): Chromatic channel of the image.
        luma_depth (ChannelDepth, optional): Channel depth of the luminance channel. Defaults to ChannelDepth.C12_BIT.
        chroma_depth (ChannelDepth, optional): Channel depth of the chromatic channel. Defaults to ChannelDepth.C8_BIT.
        rgb_depth (ChannelDepth, optional): Channel depth of the chromatic channel. Defaults to ChannelDepth.C8_BIT.

    Returns:
        tf.Tensor: Converted RGB image.
    """
    target_image_shape = tf.shape(luma)[-2:]

    # Change DataFormat from NCHW to NHWC
    # luma = change_data_format(luma, data_format_in=DataFormat.NCHW, data_format_out=DataFormat.NHWC)
    # chroma = change_data_format(chroma, data_format_in=DataFormat.NCHW, data_format_out=DataFormat.NHWC)
    luma = tf.cast(luma[tf.newaxis, :, :, tf.newaxis], tf.float32)
    chroma = tf.cast(chroma[tf.newaxis, ...], tf.float32)

    # Upscale chroma channel
    # chroma = tf.image.resize(chroma, target_image_shape, ResizeMethod.NEAREST_NEIGHBOR)

    # Preprocess luma & chroma channel between [0, 1]
    # Luminance
    left_l_value = tfp.stats.percentile(luma, 5.0)
    right_l_value = tf.reduce_max(luma)
    diff_l = right_l_value - left_l_value if (right_l_value - left_l_value) > 0 else luma_depth
    inverse_diff_l = 1. / diff_l
    l_channel = (luma - left_l_value) * inverse_diff_l
    l_channel = tf.clip_by_value(l_channel, 0., 1.)

    # Chromatic
    u_channel = (chroma[..., 0:1] / chroma_depth.value) * 0.6235
    v_channel = (chroma[..., 1:2] / chroma_depth.value) * 0.6235

    # Set out-of-gammut (u', v') to white point
    condition = tf.fill(u_channel.shape, False, tf.bool)
    condition = tf.logical_or(condition, 4.0564 * u_channel - 3.0265 * v_channel > 1)
    condition = tf.logical_or(condition, 0.2521 * u_channel + 1.6632 * v_channel > 1)
    condition = tf.logical_or(condition, -4.0272 * u_channel + 1.8027 * v_channel > 1)
    condition = tf.logical_or(condition, 4.7482 * u_channel + 2.2585 * v_channel < 1)
    condition = tf.logical_or(condition, v_channel < 0.0130)
    condition = tf.logical_or(
        condition,
        (1 - 7.7752 * u_channel - 3.6607 * v_channel + 2.0074 * u_channel * u_channel
         + 3.3520 * v_channel * v_channel + 12.8636 * u_channel * v_channel > 0)
    )
    u_channel = tf.where(condition, 0.210308593633046, u_channel)
    v_channel = tf.where(condition, 0.473759664788581, v_channel)

    # Compute X and Z coefficients (c_X, c_Z)
    coeff_in_denominator = 1.0 / (4.0 * v_channel)
    coeff_x = coeff_in_denominator * 9.0 * u_channel
    coeff_z = coeff_in_denominator * (12.0 -3.0 * u_channel - 20.0 * v_channel)

    # Compute X, Y and Z (from CIE XYZ) based on the coefficients
    y = l_channel # Y = L channel
    x = coeff_x * y
    z = coeff_z * y

    # Compute R, G and B (from RGB_linear) based on X, Y and Z
    r_linear = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    g_linear = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    b_linear = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z

    # TODO: Projection to RGB cube
    # > option 1: see documentation (standard IMP projection)
    # > option 2: see above in operator() of TLuv2RGBTrafoFunctor
    #
    # For now, just a simple cropping:
    r_linear = tf.clip_by_value(r_linear, 0., 1.)
    g_linear = tf.clip_by_value(g_linear, 0., 1.)
    b_linear = tf.clip_by_value(b_linear, 0., 1.)

    # Compute sRGB from linear RGB (i.e. standard gamma correction for sRGB)
    def srgb_gamma_correction(value: tf.Tensor) -> tf.Tensor:
        return tf.where(value <= 0.0031308, 12.92 * value, 1.055 * tf.pow(value, 1.0 / 2.4) - 0.055)

    # sRGB gamma correction
    r = tf.cast(srgb_gamma_correction(r_linear) * rgb_depth.value, dtype=tf.uint8)
    g = tf.cast(srgb_gamma_correction(g_linear) * rgb_depth.value, dtype=tf.uint8)
    b = tf.cast(srgb_gamma_correction(b_linear) * rgb_depth.value, dtype=tf.uint8)

    # Concatenate R, G and B channels
    rgb = tf.concat([r, g, b], axis=-1)

    #return shape: (B,)
    return rgb