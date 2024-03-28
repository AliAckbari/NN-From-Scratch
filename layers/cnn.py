import numpy as np

class ConvolutionalLayer:
    def __init__(self, num_filters: int, filter_size: int, input_channels: int, pad: int = 0, stride: int = 1):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.input_channels = input_channels
        self.pad = pad
        self.stride = stride
        self.filters = np.random.randn(num_filters, input_channels, filter_size, filter_size) / (filter_size * filter_size * input_channels)

    def conv_forward_naive(self, x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        A naive implementation of the forward pass for a convolutional layer.

        Inputs:
        - x: Input data of shape (N, C, H, W)
        - w: Filter weights of shape (F, C, HH, WW)
        - b: Biases, of shape (F,)

        Returns:
        - out: Output data, of shape (N, F, H', W') where H' and W' are given by
            H' = 1 + (H + 2 * pad - HH) / stride
            W' = 1 + (W + 2 * pad - WW) / stride
        """
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        _, outH, outW = self._get_output_shape(H, W, HH, WW)

        out = np.zeros((N, F, outH, outW))

        x_pad = np.pad(x, ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)), 'constant')
        
        for i in range(outH):
            for j in range(outW):
                x_region = x_pad[:, :, i * self.stride:i * self.stride + HH, j * self.stride:j * self.stride + WW]
                out[:, :, i, j] = np.sum(x_region * w, axis=(2, 3)) + b

        return out

    def conv_backward_naive(self, dout: np.ndarray, x: np.ndarray, w: np.ndarray, b: np.ndarray) -> tuple:
        """
        A naive implementation of the backward pass for a convolutional layer.

        Inputs:
        - dout: Upstream derivatives.
        - x: Input data of shape (N, C, H, W)
        - w: Filter weights of shape (F, C, HH, WW)
        - b: Biases, of shape (F,)

        Returns:
        - dx: Gradient with respect to x
        - dw: Gradient with respect to w
        - db: Gradient with respect to b
        """
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        _, outH, outW = self._get_output_shape(H, W, HH, WW)

        dx = np.zeros_like(x)
        dw = np.zeros_like(w)
        db = np.sum(dout, axis=(0, 2, 3))

        x_pad = np.pad(x, ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)), 'constant')

        for i in range(outH):
            for j in range(outW):
                x_region = x_pad[:, :, i * self.stride:i * self.stride + HH, j * self.stride:j * self.stride + WW]
                for f in range(F):
                    dw[f] += np.sum(x_region * (dout[:, f, i, j])[:, None, None, None], axis=0)
                for n in range(N):
                    dx[n, :, i * self.stride:i * self.stride + HH, j * self.stride:j * self.stride + WW] += \
                        np.sum((w[:, :, :, :] * (dout[n, :, i, j])[:, None, None, None]), axis=0)

        dx = dx[:, :, self.pad:H + self.pad, self.pad:W + self.pad]

        return dx, dw, db

    def _get_output_shape(self, H: int, W: int, HH: int, WW: int) -> tuple:
        """
        Computes the output shape after applying convolution operation.

        Inputs:
        - H: Input height
        - W: Input width
        - HH: Filter height
        - WW: Filter width

        Returns:
        - outH: Output height
        - outW: Output width
        """
        outH = 1 + (H + 2 * self.pad - HH) // self.stride
        outW = 1 + (W + 2 * self.pad - WW) // self.stride

        return outH, outW
