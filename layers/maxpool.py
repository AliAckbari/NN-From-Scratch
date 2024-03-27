import numpy as np

class MaxPool:
    def __init__(self, stride: int = 2, pool_height: int = 2, pool_width: int = 2) -> None:
        """
        Initialize the MaxPool layer with specified parameters.

        Args:
            stride (int): Stride size for pooling operation. Default is 2.
            pool_height (int): Height of the pooling window. Default is 2.
            pool_width (int): Width of the pooling window. Default is 2.
        """
        self.stride = stride
        self.pool_height = pool_height
        self.pool_width = pool_width

    def forward(self, input: np.ndarray) -> tuple:
        """
        Perform forward pass of max pooling operation.

        Args:
            input (np.ndarray): Input feature map with shape (N, C, H, W).

        Returns:
            tuple: Pooled feature map and cache for backward pass.
        """
        N, C, H, W = input.shape
        H_out = 1 + (H - self.pool_height) // self.stride
        W_out = 1 + (W - self.pool_width) // self.stride
        output = np.zeros((N, C, H_out, W_out))
        cache = []

        for i in range(N):
            for h in range(H_out):
                for w in range(W_out):
                    pool_region = input[i, :, h * self.stride:h * self.stride + self.pool_height, w * self.stride:w * self.stride + self.pool_width]
                    output[i, :, h, w] = np.amax(pool_region, axis=(1, 2))
                    # Save the pool region and its indices
                    cache.append((pool_region, np.argmax(pool_region, axis=(1, 2))))

        return output, cache
            
    def backward(self, cache: tuple, dout: np.ndarray) -> np.ndarray:
        """
        Perform backward pass of max pooling operation.
        
        Args:
            cache (tuple): Cached values from forward pass.
            dout (np.ndarray): Gradient of the loss with respect to the output.
        
        Returns:
            np.ndarray: Gradient of the loss with respect to the input.
        """
        x, pool_param = cache
        N, C, H_out, W_out = dout.shape
        H, W = x.shape[2], x.shape[3]

        # Initialize gradient
        dx = np.zeros(x.shape)
        
        # Iterate over each sample in the batch
        for i in range(N):
            # Slide the pooling window over the input feature map
            for h in range(0, H - self.pool_height + 1, self.stride):
                for w in range(0, W - self.pool_width + 1, self.stride):
                    # Extract the pooling region
                    pool_region = x[i, :, h:h + self.pool_height, w:w + self.pool_width].reshape(C, self.pool_height * self.pool_width)
                    # Find the indices of the maximum values
                    max_pool_indices = np.argmax(pool_region, axis=1)
                    # Distribute the gradient to the maximum values
                    dx[i, :, h:h + self.pool_height, w:w + self.pool_width].reshape(C, self.pool_height * self.pool_width)[np.arange(C), max_pool_indices] += dout[i].reshape(C, H_out * W_out)
                    
        return dx
