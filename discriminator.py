from processing import generate_real_samples, generate_fake_samples, preprocess_cifar10
from tensorflow.keras import layers, models, utils, optimizers


class Discriminator:
    """
    Defines and represents a discriminator model used to classify real from fake samples in a
    GAN model
    """
    def __init__(self, input_shape=(32, 32, 3)):
        self.model = models.Sequential()
        self.model._name = "cifar10_discriminator"
        model_layers = [
            # Basic Conv Layer
            layers.Conv2D(64, (3,3), padding='same', input_shape=input_shape),
            # Each of the next Convolutions is downsampling the image
            # We use LeakyReLU as activation functions
            layers.Conv2D(128, (3, 3), strides=(2,2), padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(128, (3, 3), strides=(2,2), padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(256, (3, 3), strides=(2,2), padding='same'),
            layers.LeakyReLU(alpha=0.2),
            # Classify the images
            layers.Flatten(),
            layers.Dropout(0.4),
            layers.Dense(1, activation='sigmoid')
        ]

        # Add all the layers to the model
        for layer in model_layers:
            self.model.add(layer)

        # Define the optimizer for SGD
        sgd_opt = optimizers.Adam(lr=0.0002, beta_1=0.5)

        # Compile the model
        self.model.compile(loss='binary_crossentropy', optimizer=sgd_opt, metrics=['accuracy'])


    def describe(self, filename="discriminator_plot.png"):
        """
        Displays the model summary in the terminal output and plots the model schema into the
        `filename` given as input
        """
        self.model.summary()
        utils.plot_model(self.model, to_file=filename, show_shapes=True, show_layer_names=True)


    def train(self, dataset, iterations=20, batch_size=128):
        half_batch_size = int(batch_size / 2)
        # For each given iteration
        for i in range(iterations):
            # Get real samples
            x_real, y_real = generate_real_samples(dataset, half_batch_size)
            # Train discriminator on real samples
            _, real_acc = self.model.train_on_batch(x_real, y_real)
            # Generate fake samples
            x_fake, y_fake = generate_fake_samples(half_batch_size)
            # Train discriminator on fake samples
            _, fake_acc = self.model.train_on_batch(x_fake, y_fake)

            # Summarize performance
            print('>%d real=%.0f%% fake=%.0f%%' % (i+1, real_acc*100, fake_acc*100))



if __name__ == "__main__":
    discriminator = Discriminator()
    dataset = preprocess_cifar10()
    discriminator.train(dataset)
    #discriminator.describe()

