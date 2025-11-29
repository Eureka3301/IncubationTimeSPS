import numpy as np

from filemanager import *
from visualizers import *


# all data is loaded on Mac from yandex disk by folders directly in the Downloads

comp = 'win'

if comp == 'win':
    propfilename = load_prop('win_propfiles.json')
    datafilename = load_prop('win_datafiles.json')
else:
    propfilename = load_prop('mac_propfiles.json')
    datafilename = load_prop('mac_datafiles.json')    

materials = list(datafilename.keys())
print(materials)

mat = '50mm'

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
fig, axes = LSM_SPS_visualize(xx, yy, sig_st, E, search_sig_cr, search_tau, 20, 100, 1, 0.9)

# Gather several materials
# gather(propfilename, datafilename, materials[:])
