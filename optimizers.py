import numpy as np
import time
from mechanics import *

def refine_grid(opti_i, opti_j, search_p1, search_p2, refinement_factor=0.1):
    '''
    Refine grid search around initial optimum
    '''

    p1_num = len(search_p1)
    p2_num = len(search_p2)

    p1_range = (search_p1[-1] - search_p1[0])*refinement_factor
    p2_range = (search_p2[-1] - search_p2[0])*refinement_factor

    refined_p1_l = search_p1[opti_i] -  p1_range / 2
    refined_p1_r = search_p1[opti_i] +  p1_range / 2

    refined_p2_l = search_p2[opti_j] -  p2_range / 2
    refined_p2_r = search_p2[opti_j] +  p2_range / 2
    
    # Create refined grid for this parameter
    refined_search_p1 = np.linspace(refined_p1_l, refined_p1_r, p1_num)
    refined_search_p2 = np.logspace(np.log10(refined_p2_l), np.log10(refined_p2_r), p2_num)

    return refined_search_p1, refined_search_p2


def LSM(xx, yy, search_p1, search_p2):
    '''
    LSM search via limits
    '''
    summ_grid = np.zeros((len(search_p1), len(search_p2)))

    for i, p1 in enumerate(search_p1):
        for j, p2 in enumerate(search_p2):
            summ = 0
            for k in range(len(xx)):
                summ += (yy[k] - model(xx[k], p1, p2))**2
            summ_grid[i, j] = summ

    opti_i, opti_j = np.unravel_index(np.argmin(summ_grid), summ_grid.shape)

    return opti_i, opti_j, summ_grid

def SPS(xx, yy, search_p1, search_p2, q, M):
    '''
    SPS probability area construction
    '''

    start = time.time()

    N = len(xx)

    np1 = len(search_p1)
    np2 = len(search_p2)

    np12 = np1*np2

    dvector = np.empty((np12, N))
    
    P1, P2 = np.meshgrid(search_p1, search_p2)
    P1f= P1.flatten()
    P2f= P2.flatten()

    for ij in range(np12):
        for k in range(N):
            residual = yy[k] - model(xx[k], P1f[ij], P2f[ij])
            derivative = dmodeldp2(xx[k], P1f[ij], P2f[ij])
            dvector[ij, k] = residual * derivative

    beta = np.random.choice([-1, 1], size=(N, M-1), p=[0.5, 0.5])
    beta = np.hstack([np.ones((N, 1)), beta])

    H = np.abs((dvector @ beta))
    
    flat_grid = np.zeros(np12)

    for ij in range(np12):
        sorted_H = np.sort(H[ij])
        rank = np.searchsorted(sorted_H, H[ij, 0])
        if rank < M - q:
            flat_grid[ij] = 1  # Parameter is in confidence region

    end = time.time()
    print(f'    Time consumed: {end-start:.3f} s')

    return flat_grid.reshape(np1, np2, order='F')

def SPS___(xx, yy, search_p1, search_p2, q, M):
    '''
    SPS probability area construction
    '''
    
    start = time.time()

    N = len(xx)

    # probability area
    grid = np.zeros((len(search_p1), len(search_p2)))

    # 
    beta = np.random.choice([-1, 1], size=(M-1, N), p=[0.5, 0.5])
    beta = np.vstack([np.ones((1, N)), beta])
    
    for i, p1 in enumerate(search_p1):
        for j, p2 in enumerate(search_p2):
            # beta = np.random.choice([-1, 1], size=(M-1, N), p=[0.5, 0.5])
            # beta = np.vstack([np.ones((1, N)), beta])

            delta = np.array([yy[k] - model(xx[k], p1, p2) for k in range(N)])
            deriv = np.array([dmodeldp2(xx[k], p1, p2) for k in range(N)])
            
            H = np.abs((beta @ (delta * deriv).reshape(-1, 1)).flatten())
            
            # Sort H and find rank of H[0]
            rank = np.searchsorted(np.sort(H), H[0])
            
            # Check if H[0] is in the confidence region
            if rank < M - q:
                grid[i, j] = 1  # Parameter is in confidence region

    end = time.time()
    print(f'    Time consumed: {end-start:.3f} s')

    return grid

# demo of the module
if __name__ == "__main__":
    # Generate some test data
    true_sig_cr = 100e6  # 50 MPa
    true_tau = 50e-6     # 50 microsecond

    E = 200e9           # 200 GPa
    sig_st = 100e6      # 100 MPa

    # Generate data
    xx_unit, yy_unit = gen_data(true_sig_cr/sig_st, true_tau/tau0)
    xx_abs, yy_abs = ruffle(xx_unit, yy_unit, sig_st, E)

    # Define search ranges
    search_sig_cr = np.linspace(0.5*true_sig_cr, 1.5*true_sig_cr, 50)  # 50-150 MPa
    search_tau = np.logspace(-6, -4, 100)         # 1-100 microseconds

    from visualizers import LSM_SPS_visualize
    # Visualize LSM
    fig, axes = LSM_SPS_visualize(xx_abs, yy_abs, sig_st, E, search_sig_cr, search_tau)
