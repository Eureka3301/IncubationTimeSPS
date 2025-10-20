import os

import numpy as np

import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science','no-latex'])

from filemanager import read_csv, load_prop

from optimizers_archive import *


mech_props = {
    'E' : 50e+9,
    'sig_st' : 300e+6,
}

model_params = {
    'p1' : 1,
    'p2' : 10,
}



