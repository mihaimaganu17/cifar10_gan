from cifar10 import Cifar10
import numpy as np


def preprocess_cifar10():
    """
    Preprocesses Cifar10 training images as input for the discrimininator model
    """
    # Load the object that hold the data
    cif = Cifar10()
    # Convert training data to float
    train_x = cif.train_x.astype('float32')
    # Scale from [0, 255] pixel values to [-1, 1] values for tahn function activation
    train_x = (train_x - 127.5) / 127.5

    return train_x


def generate_real_samples(dataset, n_samples):
    """
    Generates a number of `n_samples` samples by selecting them from the Cifar10 `dataset`
    """
    # Choose random indexes
    # TODO: Replace with shuffling the dataset
    random_idxs = np.random.randint(0, dataset.shape[0], n_samples)

    # Take out the samples from the dataset
    real_samples = dataset[random_idxs]

    # Generate `real` class labels
    real_labels = np.ones((n_samples, 1))

    return real_samples, real_labels


def generate_latent_points(latent_size, n_samples):
    """
    Generate points in the latent space and return them as input for the generator model

    Parameters
    ----------
    latent_size: int
        Number of latent point to be generated for each sample
    n_samples: int
        Number of samples

    Return
    ------
    samples: 2D array with shape (n_samples, latent_size)
    """
    latent_points = np.random.randn(latent_size * n_samples)
    samples = latent_points.reshape((n_samples, latent_size))
    return samples


def generate_fake_samples(latent_size, n_samples):
    """
    Generates fake Cifa10 samples with random pixel values
    """
    # Generate random pixels
    pixels = np.random.rand(32 * 32 * 3 * n_samples)
    # Convert them for tahn activation
    fake_points = -1 + pixels * 2
    # Reshape into images
    fake_images = fake_points.reshape((n_samples, 32, 32, 3))
    # Generate `fake` class labels
    fake_labels = np.zeros((n_samples, 1))

    return fake_images, fake_labels

