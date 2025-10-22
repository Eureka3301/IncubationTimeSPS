import os

import numpy as np

import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science','no-latex'])

from filemanager import read_csv, load_prop

from optimizers import *



propfilename = {
    'AC':r'/Users/rodion/Downloads/2018 Effect of microstructure/As Cast.json',
    'T5':r'/Users/rodion/Downloads/2018 Effect of microstructure/T5.json',
    'T6':r'/Users/rodion/Downloads/2020 Mechanical behavior and texture evolution/TD/TD.json',
    'SD':r'/Users/rodion/Downloads/2020 Mechanical behavior and texture evolution/SD/SD.json'
}

datafilename = {
    'AC':r'/Users/rodion/Downloads/2018 Effect of microstructure/As Cast.csv',
    'T5':r'/Users/rodion/Downloads/2018 Effect of microstructure/T5.csv',
    'T6':r'/Users/rodion/Downloads/2020 Mechanical behavior and texture evolution/TD/TD.csv',
    'SD':r'/Users/rodion/Downloads/2020 Mechanical behavior and texture evolution/SD/SD.csv'
}

mat = 'AC'
print(f'Material {mat}')

props = load_prop(propfilename[mat])

xx, yy = read_csv(datafilename[mat])

E = props.get('E(Pa)')
sig_st = props.get('sig_st(Pa)')

# Define search ranges
search_sig_cr = np.linspace(props.get('sigcr_l(Pa)'), props.get('sigcr_r(Pa)'), props.get('sigcr_num'))
search_tau = np.logspace(np.log10(props.get('tau_l(s)')), np.log10(props.get('tau_r(s)')), props.get('tau_num'))


# Visualize LSM
#fig, axes = LSM_visualize(xx, yy, sig_st, E, search_sig_cr, search_tau)

# Visualize SPS
fig, axes = SPS_visualize(xx, yy, sig_st, E, search_sig_cr, search_tau, 10, 100)