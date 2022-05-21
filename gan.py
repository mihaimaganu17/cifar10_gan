import numpy as np
from tensorflow.keras import models, optimizers, utils

from generator import Generator, generate_fake_samples
from discriminator import Discriminator
from utils import save_samples_as_plot
from processing import generate_real_samples, generate_latent_points, preprocess_cifar10


class GAN:
    def __init__(self, g_model, d_model):
        """
        Build a new GAN logic model by concatenating a generator model and a discriminator model
        """
        self.g_model = g_model
        self.d_model = d_model
        # Make the disciminator weights not trainable as part of the GAN so that we only train
        # the generator to generate better samples
        self.d_model.trainable = False

        # Define a new model
        self.model = models.Sequential()
        # Add the generator
        self.model.add(self.g_model)
        # Add the discriminator
        self.model.add(self.d_model)

        # Define an optimizer
        opt = optimizers.Adam(lr=0.0002, beta_1=0.5)
        # Compile the model
        self.model.compile(loss='binary_crossentropy', optimizer=opt)


    def describe(self, filename="gan_plot.png"):
        """
        Describes the model structure as output in the CLI and saved it in the file given by
        `filename`
        """
        self.model.summary()
        utils.plot_model(self.model, to_file=filename, show_layer_names=True, show_shapes=True)


    def train(self, dataset, latent_dim, epochs=200, batch_size=128):
        # Compute number of batches per epoch
        batch_per_epoch = int(dataset.shape[0] / batch_size)
        # Comput half of the batch size for discriminator training
        half_batch_size = int(batch_size / 2)

        # For each epoch
        for e in range(epochs):
            # TODO: Add dataset shuffling here
            # For each batch
            for batch in range(batch_per_epoch):
                # Get real samples
                x_real, y_real = generate_real_samples(dataset, half_batch_size)
                # Get fake samples
                x_fake, y_fake = generate_fake_samples(self.g_model, latent_dim, half_batch_size)
                # Train discriminator on both set of samples, resulting in one batch
                real_loss, real_acc = self.d_model.train_on_batch(x_real, y_real)
                fake_loss, fake_acc = self.d_model.train_on_batch(x_fake, y_fake)

                # Prepare latent points as input for the gan model
                x_gan = generate_latent_points(latent_dim, batch_size)
                # Create inverted labels so that our generator is updated
                y_gan = np.ones((batch_size, 1))
                # Train the model, resulting in updating the generator via the discriminators error
                g_loss = self.model.train_on_batch(x_gan, y_gan)

                # Summarize loss for this batch
                print(">%d, %d/%d, real_loss=%.3f, fake_loss=%.3f, gan_loss=%3f" %
                    (e+1, batch+1, batch_per_epoch, real_loss, fake_loss, g_loss))
            # Evaluate performance every 10th epoch
            if (e+1) % 10 == 0:
                self.summarize_performance(e, dataset, latent_dim)


    def summarize_performance(self, epoch, dataset, latent_dim, n_samples=100):
        """
        Evaluates the performance of the discriminator, plots generated images and saves
        the generator model for `epoch`
        """
        # Generate real samples
        x_real, y_real = generate_real_samples(dataset, n_samples)
        # Evaluate discriminator on real samples
        _, real_acc = self.d_model.evaluate(x_real, y_real, verbose=0)
        # Generate fake samples
        x_fake, y_fake = generate_fake_samples(self.g_model, latent_dim, n_samples)
        # Evaluate discriminator on real samples
        _, fake_acc = self.d_model.evaluate(x_fake, y_fake, verbose=0)

        # Summarize performance
        print(">Accuracy: Real samples -> %.0f%%, Fake samples -> %.0f%%" %
                (real_acc * 100, fake_acc * 100))

        # Generate filename for plot
        filename = f"generated_plot_e{epoch:03d}.png"
        # Save plot of generated samples
        save_samples_as_plot(x_fake, filename=filename)

        # Generate filename for model
        filename = f"generated_model_e{epoch:03d}.h5"
        # Save model
        self.g_model.save(filename)


if __name__ == "__main__":
    latent_size = 100
    # Fetch generator
    gen = Generator(latent_size)
    # Fetch discriminator
    disc = Discriminator()
    # Construct a new GAN model
    gan = GAN(gen.model, disc.model)

    dataset = preprocess_cifar10()

    gan.train(dataset, latent_size)
