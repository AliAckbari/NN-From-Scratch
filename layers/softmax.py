import numpy as np

class SoftMaxLayer(object):
    def __init__(self):
        """
        Initialize the Softmax layer.
        """
        self.inp = None
        self.output = None

    def forward(self, X):
        """
        Perform forward pass through the Softmax function.

        Args:
            X (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Output of the Softmax function.
        """
        self.inp = X
        exps = np.exp(X - np.max(X, axis=1, keepdims=True))
        self.output = exps / np.sum(exps, axis=1, keepdims=True)
        return self.output

    def backward(self, up_grad):
        """
        Perform backward pass through the Softmax function.

        Args:
            up_grad (numpy.ndarray): Gradient from the upper layer.

        Returns:
            numpy.ndarray: Gradient with respect to the input.
        """
        if self.inp is None:
            raise ValueError("Forward pass must be called before backward pass.")

        # Efficiently calculate the gradient using matrix multiplication
        self.output = np.clip(self.output, 1e-8, 1 - 1e-8)  # Clipping to avoid division by zero
        identity = np.eye(self.output.shape[1])  # Identity matrix of the same shape as output
        dX = self.output * np.dot(up_grad, identity - self.output.T)

        return dX

    def step(self, optimizer):
        """
        Perform parameter update (no operation for activation functions).

        Args:
            optimizer: Optimizer object.
        """
        pass
