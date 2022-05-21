from matplotlib import pyplot


def plot_samples(samples, grid_shape=(5, 5)):
    """
    Plots the given `samples` in a grid

    Parameters
    ----------
    samples: array containing the samples
    grid_shape: tuple(int, int)
        Contains the shape of the grid to be drawn
    """

    rows, columns = grid_shape
    for i in range(rows * columns):
        pyplot.subplot(rows, columns, 1 + i)
        # Turn off axis to declutter the display
        pyplot.axis('off')
        # Show image on the plot
        pyplot.imshow(samples[i])

    # Display the plot
    pyplot.show()


def save_samples_as_plot(samples, filename="generated_plot.png", grid_shape=(5, 5)):
    """
    Plot given samples and saved them in the specified `filename`
    """
    # Scale sample from [-1, 1] to [0, 1]
    samples = (1 + samples) / 2.0

    # TODO: Refactor this into separate function
    rows, columns = grid_shape
    for i in range(rows * columns):
        pyplot.subplot(rows, columns, 1 + i)
        # Turn off axis to declutter the display
        pyplot.axis('off')
        # Show image on the plot
        pyplot.imshow(samples[i])

    pyplot.savefig(filename)
    pyplot.close()
