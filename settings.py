import os

IMG_DIR = os.path.join('img')
if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR)

WEIGHT_INITIALIZERS = [
    'uniform', 'normal', 'zeros', 'ones', 'glorot_normal', 'glorot_uniform',
    'he_normal', 'he_uniform'
]
