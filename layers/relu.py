import numpy as np

class RelU:
    def __init__(self):
        """
        Initialize the Rectified Linear Unit (ReLU) activation function.
        """
        self.inp = None

    def forward(self, inp):
        """
        Perform forward pass through the ReLU function.

        Args:
            inp (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Output of the ReLU function.
        """
        self.output = np.maximum(inp, 0)
        return self.output

    def backward(self, up_grad):
        """
        Perform backward pass through the ReLU function.

        Args:
            up_grad (numpy.ndarray): Gradient from the upper layer.

        Returns:
            numpy.ndarray: Gradient with respect to the input.
        """
        # Check if self.inp is not None before using it
        if self.inp is not None:
            down_grad = up_grad * (self.inp >= 0)  # Use >= for ReLU activation
        else:
            down_grad = up_grad * 0  # Handle the case where self.inp is None (set gradient to 0)
        return down_grad

    def step(self, optimizer):
        """
        Perform parameter update (no operation for activation functions).

        Args:
            optimizer: Optimizer object.
        """
        pass
