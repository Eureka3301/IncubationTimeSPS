import numpy as np

tau0 = 1e-6 #mus

def model(X, p1, p2):
    '''
    model in unit values
    '''
    if X < p1 / p2:
        return p1 + p2 * X
    else:
        return 2 * np.sqrt(p1 * p2 * X)

def abs_model(X, sig_cr, tau, sig_st, E):
    '''
    model in absolute values
    '''
    if E * X * tau < 2 * sig_cr:
        return sig_cr + 1/2 * E * X * tau
    else:
        return np.sqrt(2 * sig_cr * E * X * tau)

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

def comb(xx, yy, sig_st, E):
    '''
    absolute values to unit data
    '''
    eps_st = sig_st/E
    rate_st = 2 * eps_st/tau0

    return xx/rate_st, yy/sig_st

def ruffle(xx, yy, sig_st, E):
    '''
    unit data to absolute values
    '''
    eps_st = sig_st/E
    rate_st = 2 * eps_st/tau0

    return xx*rate_st, yy*sig_st

def unit_params(sig_cr, tau, sig_st, E):
    '''
    abs parameters to unit parameters
    '''
    return sig_cr / sig_st, tau / tau0

def abs_params(p1, p2, sig_st, E):
    '''
    unit parameters to abs parameters
    '''
    return p1 * sig_st, p2 * tau0

# demo of the model
if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import scienceplots
    plt.style.use(['science','no-latex'])

    # Generate data
    p1, p2 = 1.0, 0.5
    xx_unit, yy_unit = gen_data(p1, p2)

    # Convert to unit variables
    sig_st, E = 100, 20000 # MPa
    xx_abs, yy_abs = ruffle(xx_unit, yy_unit, sig_st, E)

    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Absolute plot
    ax1.scatter(xx_abs, yy_abs)
    ax1.set_xscale('log')
    ax1.grid(True, alpha = 0.5)

    ax1.set_title('Absolute Values')
    ax1.set_xlabel('X (absolute)')
    ax1.set_ylabel('Y (absolute)')

    # Unit plot
    ax2.scatter(xx_unit, yy_unit)
    ax2.set_xscale('log')
    ax2.grid(True, alpha = 0.5)

    ax2.set_title('Unit Variables')
    ax2.set_xlabel('X (unit)')
    ax2.set_ylabel('Y (unit)')

    plt.tight_layout()
    plt.show()
