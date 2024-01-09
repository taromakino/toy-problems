from enum import Enum


class Task(Enum):
    ALL = 'all'
    ERM = 'erm'
    PRETRAIN = 'pretrain'
    VAE = 'vae'
    CLASSIFY = 'classify'


class EvalStage(Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'