from abc import ABC, abstractmethod
import torch
import torchaudio.transforms as T
import numpy as np
import torchaudio.transforms as transforms
import os
import sys
import torch
import functools
import torch.utils.data as Data
import pandas as pd
from sklearn.preprocessing import StandardScaler




def MinMaxScaler(data, return_scalers=False):
    """Min Max normalizer.

    Args:
      - data: original data

    Returns:
      - norm_data: normalized data
    """
    min = np.min(data, 0)
    max = np.max(data, 0)
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    if return_scalers:
        return norm_data, min, max
    return norm_data


def MinMaxArgs(data, min, max):
    """
    Args:
        data: given data
        min: given min value
        max: given max value

    Returns:
        min-max scaled data by given min and max
    """
    numerator = data - min
    denominator = max - min
    norm_data = numerator / (denominator + 1e-7)
    return norm_data



class TsImgEmbedder(ABC):
    """
    Abstract class for transforming time series to images and vice versa
    """

    def __init__(self, device, seq_len):
        self.device = device
        self.seq_len = seq_len

    @abstractmethod
    def ts_to_img(self, signal):
        """

        Args:
            signal: given time series

        Returns:
            image representation of the signal

        """
        pass

    @abstractmethod
    def img_to_ts(self, img):
        """

        Args:
            img: given generated image

        Returns:
            time series representation of the generated image
        """
        pass


class DelayEmbedder(TsImgEmbedder):
    """
    Delay embedding transformation
    """

    def __init__(self, device, seq_len, delay, embedding):
        super().__init__(device, seq_len)
        self.delay = delay
        self.embedding = embedding
        self.img_shape = None

    def pad_to_square(self, x, mask=0):
        """
        Pads the input tensor x to make it square along the last two dimensions.
        """
        _, _, cols, rows = x.shape
        max_side = max(cols, rows)
        padding = (
            0, max_side - rows, 0, max_side - cols)  # Padding format: (pad_left, pad_right, pad_top, pad_bottom)

        # Padding the last two dimensions to make them square
        x_padded = torch.nn.functional.pad(x, padding, mode='constant', value=mask)
        return x_padded

    def unpad(self, x, original_shape):
        """
        Removes the padding from the tensor x to get back to its original shape.
        """
        _, _, original_cols, original_rows = original_shape
        return x[:, :, :original_cols, :original_rows]

    def ts_to_img(self, signal, pad=True, mask=0):

        batch, length, features = signal.shape
        #  if our sequences are of different lengths, this can happen with physionet and climate datasets
        if self.seq_len != length:
            self.seq_len = length

        x_image = torch.zeros((batch, features, self.embedding, self.embedding), dtype=signal.dtype)
        i = 0
        while (i * self.delay + self.embedding) <= self.seq_len:
            start = i * self.delay
            end = start + self.embedding
            x_image[:, :, :, i] = signal[:, start:end].permute(0, 2, 1)
            i += 1

        ### SPECIAL CASE
        if i < self.embedding and i * self.delay != self.seq_len and i * self.delay + self.embedding > self.seq_len:
            start = i * self.delay
            end = signal[:, start:].permute(0, 2, 1).shape[-1]
            # end = start + (self.embedding - 1) - missing_vals
            x_image[:, :, :end, i] = signal[:, start:].permute(0, 2, 1)
            i += 1

        # cache the shape of the image before padding
        self.img_shape = (batch, features, self.embedding, i)
        x_image = x_image.to(self.device)[:, :, :, :i]

        if pad:
            x_image = self.pad_to_square(x_image, mask)

        return x_image

    def img_to_ts(self, img):
        img_non_square = self.unpad(img, self.img_shape)

        batch, channels, rows, cols = img_non_square.shape

        reconstructed_x_time_series = torch.zeros((batch, channels, self.seq_len))

        for i in range(cols - 1):
            start = i * self.delay
            end = start + self.embedding
            reconstructed_x_time_series[:, :, start:end] = img_non_square[:, :, :, i]

        ### SPECIAL CASE
        start = (cols - 1) * self.delay
        end = reconstructed_x_time_series[:, :, start:].shape[-1]
        reconstructed_x_time_series[:, :, start:] = img_non_square[:, :, :end, cols - 1]
        reconstructed_x_time_series = reconstructed_x_time_series.permute(0, 2, 1)

        return reconstructed_x_time_series.to(self.device)
