import numpy as np

class CELoss():
    def __init__(self):
        """
        Initialization of Cross Entropy Loss.
        """
        pass

    def forward(self, pred, target):
        """
        Forward pass of Cross Entropy Loss.

        Args:
            pred (numpy.ndarray): Predicted probabilities.
            target (numpy.ndarray): Ground truth labels.

        Returns:
            float: Average cross-entropy loss across the batch.
        """
        self.yhat = pred
        self.y = target
        m = self.y.shape[0]
        
        # Clip probabilities to avoid numerical issues
        self.yhat = np.clip(self.yhat, 1e-12, 1 - 1e-12)
        
        # Calculate cross-entropy loss for each sample
        loss = -np.mean(np.log(self.yhat) * self.y, axis=1)
        return np.mean(loss)  # Average loss across the batch

    def backward(self):
        """
        Backward pass of Cross Entropy Loss.

        Returns:
            numpy.ndarray: Gradient of loss with respect to the predicted probabilities.
        """
        grad = -(self.y / self.yhat) / self.y.shape[0]  # Average gradient
        return grad
