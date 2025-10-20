import numpy as np

import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')

import random

def model(X, p1, p2):
    '''
    model in unit variables
    '''
    if X < p1 / p2:
        return p1 + p2 * X
    else:
        return 2 * np.sqrt(p1 * p2 * X)



def sig_y(rate, E, sig_cr, tau):
    '''
    model in absolute variables
    '''
    rate0 = 2 * sig_cr / (E * tau)
    if rate < rate0:
        return sig_cr + tau/2 * E * rate
    else:
        return np.sqrt(2*sig_cr * E * tau * rate)



def demo_model(**params):
    '''
    plot graphics for both absolute and unit models
    '''
    # reading params if we have some
    E = params.get('E', 100e+9)
    sig_st = params.get('sig_st', 100e+6)

    sig_cr = params.get('sig_cr', 100e+6)
    tau = params.get('tau', 10e-6)

    tau0 = params.get('tau0', 1e-6)
    eps_st = sig_st / E
    rate_st = eps_st / tau0

    p1 = sig_cr / sig_st
    p2 = tau / tau0

    num = params.get('num', 100)

    # preparing data for normalized model and real data
    xx1 = np.linspace(1e-3, 5e+3, num)
    yy1 = np.array([sig_y(xx1[i], E, sig_cr, tau) for i in range(num)])

    xx2 = xx1/2 / rate_st
    yy2 = np.array([model(xx2[i], p1, p2) for i in range(num)])

    # graphical presentation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Absolute values
    ax1.plot(xx1, yy1, 'g-')
    ax1.axvline(2*sig_cr/E/tau, linestyle='--')
    ax1.set_title('Yield Stress')
    ax1.set_xlabel('$\dot{\\varepsilon}$ (1/s)')
    ax1.set_ylabel('Yield Stress (MPa)')
    ax1.grid(True)

    label1 = f'E = {E:.0e}$\mu s$\n$\sigma_{{cr}}$ = {sig_cr*1e-6:.0f}MPa\n$\\tau$ = {tau*1e+6:.0f}$\mu s$'

    plt.text(0.6, 0.1,
        label1,
        transform=ax1.transAxes,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Unit values
    ax2.plot(xx2, yy2, 'b-', label='sin(x)')
    ax2.axvline(p1/p2, linestyle='--')
    ax2.set_title('Model')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.grid(True)

    plt.text(0.6, 0.1, f'p1 = {p1:.2f}\np2 = {p2:.2f}',
        transform=ax2.transAxes,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.show()



def gen_data(way='abs', **params):
    '''
    generating some dots on arbitrary model curve
    '''
    E = params.get('E', 10e+9)
    sig_st = params.get('sig_st', 100e+6)
    tau = params.get('tau', 10e-6)
    
    tau0 = params.get('tau0', 1e-6)
    sig_cr = sig_st

    p1 = sig_cr / sig_st
    p2 = tau / tau0

    if way == 'abs':
        xx = np.array(sorted([random.uniform(1e+1, 1e+4) for _ in range(8)]))
        yy = [sig_y(x, E, sig_cr, tau) for x in xx]
    elif way == 'unit':
        xx = np.array(sorted([random.uniform(1e-4, 1) for _ in range(8)]))
        yy = [model(x, p1, p2) for x in xx]

    return (xx, yy)



def LSM(data, search_sig_cr, search_tau, way='abs', **params):
    '''
    LSM search in given bounds
    '''
    
    E = params.get('E', 10e+9)
    sig_st = params.get('sig_st', 100e+6)
    tau0 = params.get('tau0', 1e-6)
    tau = params.get('tau', 10e-6)
    sig_cr = sig_st

    p1 = sig_cr / sig_st
    p2 = tau / tau0

    xx, yy = data


    summ_grid = np.zeros((len(search_sig_cr), len(search_tau)))
    if way == 'abs':
        for i in range(len(search_sig_cr)):
            for j in range(len(search_tau)):
                for k in range(len(xx)):
                    summ_grid[i][j] += (yy[k] - sig_y(xx[k], E, search_sig_cr[i], search_tau[j]))**2
                #print(f'summ = {summ:.3e}, opti_summ = {opti_summ:.3e}, sig_cr = {sig_cr*1e-6:.0f}, tau = {tau*1e+6:.0f}')
        opti_i, opti_j = np.unravel_index(np.argmin(summ_grid), summ_grid.shape)

        opti_sig_cr = search_sig_cr[opti_i]
        opti_tau = search_tau[opti_j]
        opti_summ = summ_grid[opti_i][opti_j]
    elif way == 'unit':
        search_p1 = search_sig_cr / sig_st
        search_p2 = search_tau / tau0
        for i in range(len(search_p1)):
            for j in range(len(search_p2)):
                for k in range(len(xx)):
                    summ_grid[i][j] += (yy[k] - model(xx[k], search_p1, search_p2))**2
        opti_i, opti_j = np.unravel_index(np.argmin(summ_grid), summ_grid.shape)

        opti_sig_cr = search_p1[opti_i] * sig_st
        opti_tau = search_p2[opti_j] * tau0
        opti_summ = summ_grid[opti_i][opti_j]
        summ_grid = summ_grid * sig_st * sig_st
    

    return (opti_sig_cr, opti_tau, summ_grid)



def demo_LSM(**params):
    '''
    plot how LSM works on arbitrary data
    '''
    data = gen_data(way='unit', **params)

    search_sig_cr = np.linspace(250+6, 350e+6, 100)
    search_tau = np.linspace(1e-6, 20e-6, 50)

    sig_cr, tau, summ_grid = LSM(data,
                        search_sig_cr,
                        search_tau,
                        'abs',
                        **params)

    
    num = 100
    E = params.get('E', 100e+9)

    # preparing data for normalized model and real data
    xx = np.linspace(1e-3, 1e+4, num)
    yy = np.array([sig_y(xx[i], params['E'], sig_cr, tau) for i in range(num)])

    # graphical presentation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Absolute values
    ax1.plot(xx, yy, 'g-')
    ax1.scatter(data[0], data[1], s=20)
    ax1.axvline(2*sig_cr/E/tau, linestyle='--')
    ax1.set_title('Yield Stress')
    ax1.set_xlabel('$\dot{\\varepsilon}$ (1/s)')
    ax1.set_ylabel('Yield Stress (MPa)')
    ax1.grid(True)

    label1 = f'E = {E*1e-9:.0f}GPa\n$\sigma_{{cr}}$ = {sig_cr*1e-6:.0f}MPa\n$\\tau$ = {tau*1e+6:.0f}$\mu s$'

    plt.text(0.6, 0.1,
        label1,
        transform=ax1.transAxes,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


    # Create meshgrid for proper plotting
    SIG_CR, TAU = np.meshgrid(search_sig_cr, search_tau, indexing='ij')
    
    # Use contourf for smooth heatmap
    contour = ax2.contourf(SIG_CR, TAU, summ_grid, levels=100, cmap='plasma')
    plt.colorbar(contour, ax=ax2, label='Sum of Squared Residuals')
    
    # Plot the real parameters and optimal found parameters
    ax2.scatter(params['sig_st'], params['tau'], color='red', s=100, label='Real Point', marker='x')
    ax2.scatter(sig_cr, tau, color='green', s=100, label='Optimal Point', marker='x')
    
    ax2.set_title('LSM Optimization Landscape')
    ax2.set_xlabel('$\sigma_{cr}$ (Pa)')
    ax2.set_ylabel('$\\tau$ (s)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()













    
def plot_sig_y(E, sig_cr, tau):
    '''
    plot graphics for absolute model
    '''
    num = 100

    # preparing data for normalized model and real data
    xx = np.linspace(1e-3, 1e+4, num)
    yy = np.array([sig_y(xx[i], E, sig_cr, tau) for i in range(num)])

    # graphical presentation
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    # Absolute values
    ax.plot(xx, yy, 'g-')
    ax.axvline(2*sig_cr/E/tau, linestyle='--')
    ax.set_title('Yield Stress')
    ax.set_xlabel('$\dot{\\varepsilon}$ (1/s)')
    ax.set_ylabel('Yield Stress (MPa)')
    ax.grid(True)

    label1 = f'E = {E*1e-9:.0f}GPa\n$\sigma_{{cr}}$ = {sig_cr*1e-6:.0f}MPa\n$\\tau$ = {tau*1e+6:.0f}$\mu s$'

    plt.text(0.6, 0.1,
        label1,
        transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
