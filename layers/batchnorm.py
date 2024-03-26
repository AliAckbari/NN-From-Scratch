import numpy as np

class BatchNorm:
    def __init__(self, num_features, epsilon=1e-5, momentum=0.9):
        """
        Batch Normalization layer initialization.
        
        Args:
            num_features (int): Number of features in the input.
            epsilon (float): Small value added to the variance to avoid division by zero.
            momentum (float): Momentum for updating running mean and variance.
        """
        self.epsilon = epsilon
        self.momentum = momentum
        self.num_features = num_features
        self.gamma = np.ones((1, num_features))  # Scaling parameter
        self.beta = np.zeros((1, num_features))  # Shift parameter
        self.running_mean = np.zeros((1, num_features))  # Running mean for inference
        self.running_var = np.zeros((1, num_features))   # Running variance for inference
        
    def forward(self, inp, training=True):
        """
        Forward pass of Batch Normalization.
        
        Args:
            inp (numpy.ndarray): Input data.
            training (bool): Flag indicating whether training or inference.
            
        Returns:
            out (numpy.ndarray): Normalized output.
        """
        if training:
            # Compute batch statistics during training
            self.batch_mean = np.mean(inp, axis=0)
            self.batch_var = np.var(inp, axis=0)
            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.batch_var
        else:
            # Use running statistics during inference
            self.batch_mean = self.running_mean
            self.batch_var = self.running_var
        
        # Normalize the input
        self.x_normalized = (inp - self.batch_mean) / np.sqrt(self.batch_var + self.epsilon)
        # Scale and shift
        self.out = self.gamma * self.x_normalized + self.beta
        return self.out
    
    def backward(self, up_grad):
        """
        Backward pass of Batch Normalization.
        
        Args:
            up_grad (numpy.ndarray): Gradient from the upper layer.
            
        Returns:
            dx (numpy.ndarray): Gradient with respect to the input.
        """
        m = up_grad.shape[0]
        dx_normalized = up_grad * self.gamma
        dvar = np.sum(dx_normalized * (self.inp - self.batch_mean) * (-0.5) * np.power(self.batch_var + self.epsilon, -1.5), axis=0)
        dmean = np.sum(dx_normalized * (-1 / np.sqrt(self.batch_var + self.epsilon)), axis=0) + dvar * np.mean(-2.0 * (self.inp - self.batch_mean), axis=0)
        dx = (dx_normalized / np.sqrt(self.batch_var + self.epsilon)) + (dvar * 2.0 * (self.inp - self.batch_mean) / m) + (dmean / m)
        
        # Compute gradients w.r.t. gamma and beta
        self.dgamma = np.sum(up_grad * self.x_normalized, axis=0)
        self.dbeta = np.sum(up_grad, axis=0)
        
        return dx