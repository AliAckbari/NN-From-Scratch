import numpy as np
class GradientDescent(object):
    def __init__(self, lr):
        """
        Initialize the Gradient Descent optimizer.

        Args:
            lr (float): Learning rate.
        """
        self.lr = lr

    def get_next_update(self, x, dx):
        """
        Update the parameters using Gradient Descent.

        Args:
            x (numpy.ndarray): Current parameters.
            dx (numpy.ndarray): Gradient of parameters.

        Returns:
            numpy.ndarray: Updated parameters.
        """
        # Compute the new value for 'x' and return the result
        x = x - self.lr * dx
        return x

class Adam(object):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initialize the Adam optimizer.

        Args:
            lr (float): Learning rate.
            beta1 (float): Exponential decay rate for the first moment estimates.
            beta2 (float): Exponential decay rate for the second moment estimates.
            epsilon (float): Small value added to avoid division by zero.
        """
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment estimate
        self.v = None  # Second moment estimate
        self.t = 0     # Time step

    def get_next_update(self, x, dx):
        """
        Update the parameters using Adam.

        Args:
            x (numpy.ndarray): Current parameters.
            dx (numpy.ndarray): Gradient of parameters.

        Returns:
            numpy.ndarray: Updated parameters.
        """
        self.t += 1
        if self.m is None:
            self.m = np.zeros_like(x)
            self.v = np.zeros_like(x)
        self.m = self.beta1 * self.m + (1 - self.beta1) * dx
        self.v = self.beta2 * self.v + (1 - self.beta2) * (dx ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        x -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return x
