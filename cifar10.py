from matplotlib import pyplot
from tensorflow.keras.datasets import cifar10


class Cifar10:
    def __init__(self):
        # Load images into memory
        (self.train_x, self.train_y), (self.test_x, self.test_y) = cifar10.load_data()


    def summary(self):
        # Print the overall shape of the dataset
        print("Train: ", self.train_x.shape, self.train_y.shape)
        print("Test: ", self.test_x.shape, self.test_y.shape)


    def image_demo(self, n=5):
        """
        Plots a few images from the dataset over an `n`x`n` grid
        Parameters
        ----------
        n: int
            Size of the grip on rows and columns
        """
        for i in range(n*n):
            pyplot.subplot(n, n, i+1)
            pyplot.axis("off")
            pyplot.imshow(self.train_x[i])
        pyplot.show()

