from tensorflow.keras import models, layers, utils
from processing import generate_latent_points
from utils import plot_samples

import numpy as np


class Generator:
    def __init__(self, input_dim):
        self.model = models.Sequential()
        self.model._name = "cifar10_generator"

        # Define the size of the initial generated image. It is basically 256 filters, each of 
        # size 4x4
        initial_filter_size = 4 * 4
        n_filters = 256
        initial_dim = n_filters * initial_filter_size

        # Define all the layers for the model
        model_layers = [
            # First hidden layer
            layers.Dense(initial_dim, input_dim=input_dim),
            # Add activation
            layers.LeakyReLU(alpha=0.2),
            # Reshape it into 256 feature maps, each of size 4x4
            layers.Reshape((4, 4, 256)),
            # Upsample the 4x4 filters to 8x8 filters.
            layers.Conv2DTranspose(128, (4, 4), strides=(2,2), padding='same'),
            layers.LeakyReLU(alpha=0.2),
            # Upsample the 8x8 filters to 16x16 filters.
            layers.Conv2DTranspose(128, (4, 4), strides=(2,2), padding='same'),
            layers.LeakyReLU(alpha=0.2),
            # Upsample the 16x16 filters to 32x32 filters.
            layers.Conv2DTranspose(128, (4, 4), strides=(2,2), padding='same'),
            layers.LeakyReLU(alpha=0.2),
            # Output layer will have 3 filters, one for each color channel and will preserve their
            # 32x32 size as the original Cifar10 images are
            layers.Conv2D(3, (3, 3), activation='tanh', padding='same')
        ]

        # Add all the layers to the model
        for layer in model_layers:
            self.model.add(layer)


    def describe(self, filename='generator_plot.png'):
        self.model.summary()
        utils.plot_model(self.model, to_file=filename, show_shapes=True, show_layer_names=True)


def generate_fake_samples(g_model, latent_dim, n_samples):
    """
    Generates fake samples using latent points and the generator model prediction
    """
    input_samples = generate_latent_points(latent_dim, n_samples)
    # Generate based on latent points given as input
    generated_samples = g_model.predict(input_samples)
    # Create `fake` labels
    fake_labels = np.zeros((n_samples, 1))
    return generated_samples, fake_labels


if __name__ == "__main__":
    latent_size = 100
    generator = Generator(latent_size)
    samples, _ = generator.fake_samples(latent_size, 100)
    # Scale values from [-1, 1] to pixel value [0, 1]
    samples = (samples + 1) / 2.0
    plot_samples(samples)
