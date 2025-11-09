import numpy as np

from filemanager import *
from mechanics import *
from optimizers import *

from tqdm import tqdm

import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science','no-latex'])



def gather(propfilename, datafilename, mat_names):
    '''
    list of properties
    list of exp data
    list of material names
    Does LSM and plots model fit for all in one axes
    '''
    
    plt.ion()
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    plt.draw()
    plt.pause(0.1)

    for mat in mat_names:
        # all series properties
        props = load_prop(propfilename[mat])

        # experimental data in absolute values
        xx, yy = read_csv(datafilename[mat])

        # mechanical properties
        E = props.get('E(Pa)')
        sig_st = props.get('sig_st(Pa)')

        # search ranges in absolute parameters
        search_sig_cr = np.linspace(props.get('sigcr_l(Pa)'), props.get('sigcr_r(Pa)'), props.get('sigcr_num'))
        search_tau = np.logspace(np.log10(props.get('tau_l(s)')), np.log10(props.get('tau_r(s)')), props.get('tau_num'))

        # search ranges to unit parameters
        search_p1, search_p2 = unit_params(search_sig_cr, search_tau, sig_st, E)

        # experimental data in unit values
        xx_unit, yy_unit = comb(xx, yy, sig_st, E)

        # optimal unit parameters (LSM)
        opti_i, opti_j, grid = LSM(xx_unit, yy_unit, search_p1, search_p2)
        p1 = search_p1[opti_i]
        p2 = search_p2[opti_j]
        
        # optimal absolute parameters (LSM)
        sig_cr, tau = abs_params(p1, p2, sig_st, E)

        # unit model curve
        # x_curve_unit = np.logspace(np.log10(1e-8), np.log10(10), 100)
        # y_curve_unit = np.array([model(x, p1, p2) for x in x_curve_unit])
        # absolute model curve
        # x_curve_abs, y_curve_abs = ruffle(x_curve_unit, y_curve_unit, sig_st, E)

        # absolute model curve
        x_curve_abs = np.logspace(np.log10(1e1), np.log10(1e4), 100)
        y_curve_abs = np.array([abs_model(x, sig_cr, tau, sig_st, E) for x in x_curve_abs])
    
        label_sig_cr = f'$\\sigma_{{cr}}$ = {p1*sig_st/1e6:.0f}$MPa$, '
        label_sig_st = f'$\\sigma_{{st}}$ = {sig_st/1e6:.0f}$MPa$, '
        label_tau = f'$\\tau$ = {p2:.1f}$\\mu s$'
        label_params = label_sig_st + label_tau

        ax.scatter(xx, yy/1e6, label=f'{mat}', s=15)
        ax.plot(x_curve_abs, y_curve_abs/1e6, label=label_params)

        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)

        ax.set_title('Data and Model Fit')
        ax.set_xlabel('Strain rate (1/s)')
        ax.set_ylabel('Stress (MPa)')
        ax.legend()
        
        # Console Results
        print(f"{mat}")
        print(f"    E = {E:.1e} and sig_st = {sig_st:.1e}")

        print("Optimal parameters:")
        print(f"    sig_cr = {sig_cr/1e6:.0f} MPa")
        print(f"    tau = {tau:.2e} s")

        print("Unit parameters:")
        print(f"    p1 = {p1:.2f}, p2 = {p2:.2f}")

        plt.draw()
        plt.pause(0.1)

    plt.tight_layout()
    plt.ioff()
    plt.show()


# demo of the module
if __name__ == "__main__":

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
        'AC+',
        'AC',
        'SD',
        'T5',
        'T6',
        'Mg',
        'MgGd',
        'MgCa',
        '6.5Gd',
        '7.5Gd',
        '8.5Gd'
    ]

    gather(propfilename, datafilename, materials[:8])
