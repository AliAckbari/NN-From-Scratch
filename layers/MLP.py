class MLP:
    def __init__(self, layers, loss_fn, optimizer):
        """
        Initialize the Multi-Layer Perceptron (MLP).

        Args:
            layers (list): List of layers composing the MLP.
            loss_fn: Loss function used for training.
            optimizer: Optimization algorithm for updating weights.
        """
        self.layers = layers
        self.losses = []  # List to store losses during training
        self.loss_fn = loss_fn  # Loss function object
        self.optimizer = optimizer  # Optimizer object

    def forward(self, inp):
        """
        Forward pass through the MLP.

        Args:
            inp (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Output of the MLP.
        """
        for layer in self.layers:
            inp = layer.forward(inp)
        return inp

    def loss(self, pred, label):
        """
        Calculate the loss.

        Args:
            pred (numpy.ndarray): Predicted output.
            label (numpy.ndarray): Ground truth label.

        Returns:
            float: Loss value.
        """
        loss = self.loss_fn.forward(pred, label)
        return loss

    def backward(self):
        """
        Backward pass through the MLP.

        Returns:
            numpy.ndarray: Gradient with respect to the input.
        """
        grad = self.loss_fn.backward()
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def update(self):
        """
        Update the weights of the MLP using the optimizer.
        """
        for layer in self.layers:
            layer.step(self.optimizer)
