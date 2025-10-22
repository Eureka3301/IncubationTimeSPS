import numpy as np

from filemanager import *
from mechanics import *
from optimizers import *
from collection import *

import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science','no-latex'])

# all data is loaded on Mac from yandex disk by folders directly in the Downloads

propfilename = {
'AC':r'/Users/rodion/Downloads/2018 Effect of microstructure/As Cast.json',
'AC+':r'/Users/rodion/Downloads/2018 Effect of microstructure/As Cast and SD.json',
'T5':r'/Users/rodion/Downloads/2018 Effect of microstructure/T5.json',
'T6':r'/Users/rodion/Downloads/2020 Mechanical behavior and texture evolution/TD/TD.json',
'SD':r'/Users/rodion/Downloads/2020 Mechanical behavior and texture evolution/SD/SD.json',
'MgGd':r'/Users/rodion/Downloads/Оцифровка наших кривых/MgGd.json',
'MgCa':r'/Users/rodion/Downloads/Оцифровка наших кривых/MgCa.json',
'Mg':r'/Users/rodion/Downloads/Оцифровка наших кривых/Mg.json'
}

datafilename = {
    'AC':r'/Users/rodion/Downloads/2018 Effect of microstructure/As Cast.csv',
    'AC+':r'/Users/rodion/Downloads/2018 Effect of microstructure/As Cast and SD.csv',
    'T5':r'/Users/rodion/Downloads/2018 Effect of microstructure/T5.csv',
    'T6':r'/Users/rodion/Downloads/2020 Mechanical behavior and texture evolution/TD/TD.csv',
    'SD':r'/Users/rodion/Downloads/2020 Mechanical behavior and texture evolution/SD/SD.csv',
    'MgGd':r'/Users/rodion/Downloads/Оцифровка наших кривых/MgGd.csv',
    'MgCa':r'/Users/rodion/Downloads/Оцифровка наших кривых/MgCa.csv',
    'Mg':r'/Users/rodion/Downloads/Оцифровка наших кривых/Mg.csv'
}

materials = [
    'AC',
    'SD',
    'AC+',
    'T5',
    'T6',
    'Mg',
    'MgGd',
    'MgCa'
]

mat = 'AC'

props = load_prop(propfilename[mat])

xx, yy = read_csv(datafilename[mat])

E = props.get('E(Pa)')
sig_st = props.get('sig_st(Pa)')

# Define search ranges
search_sig_cr = np.linspace(props.get('sigcr_l(Pa)'), props.get('sigcr_r(Pa)'), props.get('sigcr_num'))
search_tau = np.logspace(np.log10(props.get('tau_l(s)')), np.log10(props.get('tau_r(s)')), props.get('tau_num'))

# Visualize LSM
# fig, axes = LSM_visualize(xx, yy, sig_st, E, search_sig_cr, search_tau)

# Visualize SPS
fig, axes = SPS_visualize(xx, yy, sig_st, E, search_sig_cr, search_tau, 10, 100)

# Gather several materials
# gather(propfilename, datafilename, materials[:])
