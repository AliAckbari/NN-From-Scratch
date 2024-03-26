import numpy as np

class Linear:
    def __init__(self, in_dim, out_dim):
        """
        Initialize the Linear layer with random weights and biases.

        Args:
            in_dim (int): Input dimension.
            out_dim (int): Output dimension.
        """
        self.in_feat = in_dim
        self.out_feat = out_dim
        # Initialize weights using He initialization
        self.weights = np.random.randn(self.in_feat, self.out_feat) * np.sqrt(2.0 / (self.in_feat + self.out_feat))
        self.bias = np.zeros(self.out_feat)
        
    def forward(self, inp):
        """
        Forward pass of the Linear layer.

        Args:
            inp (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Output of the linear transformation.
        """
        self.inp = inp
        self.Z = np.dot(inp, self.weights) + self.bias
        return self.Z
    
    def backward(self, up_grad):
        """
        Backward pass of the Linear layer.

        Args:
            up_grad (numpy.ndarray): Gradient from the upper layer.

        Returns:
            numpy.ndarray: Gradient with respect to the input.
        """
        self.dweights = np.dot(up_grad.T, self.inp)
        self.dbias = np.sum(up_grad, axis=0)
        dX = np.dot(up_grad, self.weights.T)
        return dX

    def step(self, optimizer):
        """
        Update weights and biases using the given optimizer.

        Args:
            optimizer: Optimizer object.
        """
        if self.bias is not None:
            # Update weights and biases if they exist
            self.weights, self.bias = optimizer.get_next_update(self.weights, self.dweights.T), \
                                      optimizer.get_next_update(self.bias, self.dbias)
        else:
            # Update weights only if no biases are present
            self.weights = optimizer.get_next_update(self.weights, self.dweights)
