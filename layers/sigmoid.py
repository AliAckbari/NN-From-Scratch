import numpy as np

class Sigmoid:
    def __init__(self):
        """
        Initialize the Sigmoid activation function.
        """
        self.out = None

    def forward(self, inp):
        """
        Perform forward pass through the Sigmoid function.

        Args:
            inp (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Output of the Sigmoid function.
        """
        self.out = 1.0 / (1.0 + np.exp(-inp))
        return self.out

    def backward(self, up_grad):
        """
        Perform backward pass through the Sigmoid function.

        Args:
            up_grad (numpy.ndarray): Gradient from the upper layer.

        Returns:
            numpy.ndarray: Gradient with respect to the input.
        """
        if self.out is None:
            raise ValueError("Forward pass must be called before backward pass.")
        down_grad = up_grad * self.out * (1.0 - self.out)
        return down_grad

    def step(self, optimizer):
        """
        Perform parameter update (no operation for activation functions).

        Args:
            optimizer: Optimizer object.
        """
        pass
