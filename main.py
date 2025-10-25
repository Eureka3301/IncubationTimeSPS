import numpy as np

from filemanager import *
from visualizers import *


# all data is loaded on Mac from yandex disk by folders directly in the Downloads

propfilename = {
    'AC':r'C:\Users\rodio\Downloads\2018 Effect of microstructure/As Cast.json',
    'AC+':r'C:\Users\rodio\Downloads\2018 Effect of microstructure/As Cast and SD.json',
    'T5':r'C:\Users\rodio\Downloads\2018 Effect of microstructure/T5.json',
    'T6':r'C:\Users\rodio\Downloads\2020 Mechanical behavior and texture evolution/TD/TD.json',
    'SD':r'C:\Users\rodio\Downloads\2020 Mechanical behavior and texture evolution/SD/SD.json',
    'MgGd':r'C:\Users\rodio\Downloads\Оцифровка наших кривых/MgGd.json',
    'MgCa':r'C:\Users\rodio\Downloads\Оцифровка наших кривых/MgCa.json',
    'Mg':r'C:\Users\rodio\Downloads\Оцифровка наших кривых/Mg.json',
    '6.5Gd':r'C:\Users\rodio\Downloads\2024 Influence of Gd Content/6.5 Gd.json',
    '7.5Gd':r'C:\Users\rodio\Downloads\2024 Influence of Gd Content/7.5 Gd.json',
    '8.5Gd':r'C:\Users\rodio\Downloads\2024 Influence of Gd Content/8.5 Gd.json'
    }

datafilename = {
    'AC':r'C:\Users\rodio\Downloads\2018 Effect of microstructure/As Cast.csv',
    'AC+':r'C:\Users\rodio\Downloads\2018 Effect of microstructure/As Cast and SD.csv',
    'T5':r'C:\Users\rodio\Downloads\2018 Effect of microstructure/T5.csv',
    'T6':r'C:\Users\rodio\Downloads\2020 Mechanical behavior and texture evolution/TD/TD.csv',
    'SD':r'C:\Users\rodio\Downloads\2020 Mechanical behavior and texture evolution/SD/SD.csv',
    'MgGd':r'C:\Users\rodio\Downloads\Оцифровка наших кривых/MgGd.csv',
    'MgCa':r'C:\Users\rodio\Downloads\Оцифровка наших кривых/MgCa.csv',
    'Mg':r'C:\Users\rodio\Downloads\Оцифровка наших кривых/Mg.csv',
    '6.5Gd':r'C:\Users\rodio\Downloads\2024 Influence of Gd Content/6.5 Gd.csv',
    '7.5Gd':r'C:\Users\rodio\Downloads\2024 Influence of Gd Content/7.5 Gd.csv',
    '8.5Gd':r'C:\Users\rodio\Downloads\2024 Influence of Gd Content/8.5 Gd.csv'
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
fig, axes = LSM_SPS_visualize(xx, yy, sig_st, E, search_sig_cr, search_tau, 20, 100, 2, 0.9)

# Gather several materials
# gather(propfilename, datafilename, materials[:])
