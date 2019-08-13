import os

FIGURES_DIR = os.path.join('figures')
if not os.path.exists(FIGURES_DIR):
    os.makedirs(FIGURES_DIR)

WEIGHT_INITIALIZERS = ['uniform', 'normal', 'zeros', 'ones', 'glorot_normal',
                        'glorot_uniform', 'he_normal', 'he_uniform']
