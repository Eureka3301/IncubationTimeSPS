from filemanager import read_csv
import numpy as np

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

def nomagicthegathering(exp_data, models, Es, sig_sts):

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    for i in range(len(exp_data)):
        xx = exp_data[i][0]
        yy = exp_data[i][1]
        mat = exp_data[i][2]
        ax.scatter(xx, yy, label=f'Experimental data {mat}')

    
    x_curve = np.logspace(np.log10(1e-8), np.log10(10), 100)
    for j in range(len(models)):
        p1 = models[j][0]
        p2 = models[j][1]
        y_curve = np.array([model(x, p1, p2) for x in x_curve])
    
        E = Es[j]
        sig_st = sig_sts[j]
        # Convert back to absolute values for plotting
        x_curve_abs, y_curve_abs = comeback(x_curve, y_curve, sig_st, E)
    
        label_params = f'$\sigma_{{cr}}$={p1*sig_st/1e6:.0f}$MPa$,\n$\\tau$={p2*tau0:.1e}$\mu s$'
        ax.plot(x_curve_abs, y_curve_abs, label=f'Model LSM fit\n'+label_params)
    
    ax.set_xlabel('Strain rate (1/s)')
    ax.set_ylabel('Stress (MPa)')
    ax.set_title('Data and Model Fit')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    plt.tight_layout()
    plt.show()


# demo of the module
if __name__ == "__main__":
    datafilename = {
    'AC':r'C:\Users\rodio\Downloads\Data\As Cast.csv',
    'T':r'C:\Users\rodio\Downloads\Data\T5.csv'
    }

    exp_data = [(*read_csv(datafilename['AC']), 'AC'), (*read_csv(datafilename['T']),'T')]

    Es = [39e9, 45e9]

    sig_sts = [139e6, 194e6]

    # search in area of sig_st
    # models = [(143/139, 0.6075292), (200/194, 0.6280291)]

    # arbitrary search
    models = [(155/139, 0.5145444), (246/194, 0.3678380)]

    nomagicthegathering(exp_data, models, Es, sig_sts)