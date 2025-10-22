from filemanager import read_csv
import numpy as np
from tqdm import tqdm


from filemanager import read_csv, load_prop

import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science','no-latex'])

def model(X, p1, p2):
    '''
    model in unit variables
    '''
    if X < p1 / p2:
        return p1 + p2 * X
    else:
        return 2 * np.sqrt(p1 * p2 * X)

def dmodeldp2(X, p1, p2):
    '''
    model derivative by p2
    '''
    if X < p1 / p2:
        return X
    else:
        return np.sqrt(p1*p2/X)

def gen_data(p1, p2):
    '''
    generates model dots in random spots
    '''
    left_b = 1e-3
    right_b = 10
    dot_num = 8

    xx = np.exp(np.random.uniform(np.log(left_b), np.log(right_b), dot_num))
    yy = np.array([model(x, p1, p2) for x in xx])

    return xx, yy

tau0 = 1e-6 #mus

def prepare(xx, yy, sig_st, E):
    '''
    prepares exp data for unit values
    '''
    eps_st = sig_st/E
    rate_st = eps_st/(2*tau0)

    return xx/rate_st, yy/sig_st

def comeback(xx, yy, sig_st, E):
    '''
    unit data is returned to absolute values
    '''
    eps_st = sig_st/E
    rate_st = eps_st/(2*tau0)

    return xx*rate_st, yy*sig_st

def LSM(xx, yy, search_p1, search_p2):
    '''
    LSM search via limits
    '''
    summ_grid = np.zeros((len(search_p1), len(search_p2)))

    for i, p1 in tqdm(enumerate(search_p1), total=len(search_p1)):
        for j, p2 in enumerate(search_p2):
            summ = 0
            for k in range(len(xx)):
                summ += (yy[k] - model(xx[k], p1, p2))**2
            summ_grid[i, j] = summ

    opti_i, opti_j = np.unravel_index(np.argmin(summ_grid), summ_grid.shape)

    return opti_i, opti_j, summ_grid


def gather(propfilename, datafilename, mats):
 
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    for mat in mats:
        props = load_prop(propfilename[mat])

        xx, yy = read_csv(datafilename[mat])

        E = props.get('E(Pa)')
        sig_st = props.get('sig_st(Pa)')
        print(f'E = {E} and sig_st = {sig_st}')

        # Define search ranges
        search_sig_cr = np.linspace(props.get('sigcr_l(Pa)'), props.get('sigcr_r(Pa)'), props.get('sigcr_num'))
        search_tau = np.logspace(np.log10(props.get('tau_l(s)')), np.log10(props.get('tau_r(s)')), props.get('tau_num'))

        ax.scatter(xx, yy, label=f'Experimental data {mat}')

        # Convert search ranges to unit parameters
        search_p1 = search_sig_cr / sig_st
        search_p2 = search_tau / tau0

        xx_unit, yy_unit = prepare(xx, yy, sig_st, E)

        opti_i, opti_j, grid = LSM(xx_unit, yy_unit, search_p1, search_p2)
        p1 = search_p1[opti_i]
        p2 = search_p2[opti_j]
        
        # Convert grid parameters to absolute values for contour labels
        sig_cr = p1 * sig_st
        tau = p2 * tau0

        x_curve = np.logspace(np.log10(1e-8), np.log10(10), 100)
        y_curve = np.array([model(x, p1, p2) for x in x_curve])
    
        # Convert back to absolute values for plotting
        x_curve_abs, y_curve_abs = comeback(x_curve, y_curve, sig_st, E)
    
        label_params = f'$\sigma_{{cr}}$={p1*sig_st/1e6:.0f}$MPa$, $\sigma_{{st}}$={sig_st/1e6:.0f}$MPa$,\n$\\tau$={p2*tau0:.1e}$\mu s$'
        ax.plot(x_curve_abs, y_curve_abs, label=f'Model LSM fit\n'+label_params)
        
        ax.set_xlabel('Strain rate (1/s)')
        ax.set_ylabel('Stress (MPa)')
        ax.set_title('Data and Model Fit')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')

        # Print optimal parameters
        print(f"{mat} Optimal parameters:")
        print(f"  sig_cr = {sig_cr/1e6:.0f} MPa")
        print(f"  tau = {tau:.2e} s")
        print(f"  Unit parameters: p1 = {p1:.2f}, p2 = {p2:.2f}")

    plt.tight_layout()
    plt.show()


# demo of the module
if __name__ == "__main__":

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
        'AC+',
        'T5',
        'T6',
        'Mg',
        'MgGd',
        'MgCa'
    ]

    gather(propfilename, datafilename, materials)



