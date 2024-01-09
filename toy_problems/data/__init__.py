N_CLASSES = 2
N_ENVS = 3


import data.cmnist
import data.dsprites


MAKE_DATA = {
    'cmnist': data.cmnist.make_data,
    'dsprites': data.dsprites.make_data
}