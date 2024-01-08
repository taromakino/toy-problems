import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


sns.set_context(context='paper', font_scale=2)


def remove_ticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])


def next_color(ax):
    return next(ax._get_lines.prop_cycler)['color']


def plot_red_green_image(ax, image):
    '''
    Input image has shape (2, m, n)
    '''
    _, m, n = image.shape
    image = image.transpose((1, 2, 0))
    image = np.dstack((image, np.zeros((m, n))))
    ax.imshow(image)