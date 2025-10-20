import os

import numpy as np

import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science','no-latex'])

from filemanager import read_csv, load_prop

from optimizers import *



propfilename = {
    'AC':r'C:\Users\rodio\Downloads\Data\As Cast.json',
    'T':r'C:\Users\rodio\Downloads\Data\T5.json'
}

datafilename = {
    'AC':r'C:\Users\rodio\Downloads\Data\As Cast.csv',
    'T':r'C:\Users\rodio\Downloads\Data\T5.csv'
}

mat = 'T'
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

fig, axes = SPS_visualize(xx, yy, sig_st, E, search_sig_cr, search_tau, 10, 100)