import numpy as np
import random
from tqdm import tqdm

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

def SPS(xx, yy, search_p1, search_p2, q, M):
    '''
    SPS probability area construction
    '''
    N = len(xx)

    # SPS parameters
    prob = 1 - q/M

    # probability area - initialize grid properly
    grid = np.zeros((len(search_p1), len(search_p2)))

    for i, p1 in tqdm(enumerate(search_p1), total=len(search_p1)):
        for j, p2 in enumerate(search_p2):
            beta = np.random.choice([-1, 1], size=(M-1, N), p=[0.5, 0.5])
            beta = np.vstack([np.ones((1, N)), beta])

            delta = np.array([yy[k] - model(xx[k], p1, p2) for k in range(N)])
            deriv = np.array([dmodeldp2(xx[k], p1, p2) for k in range(N)])
            
            H = np.abs((beta @ (delta * deriv).reshape(-1, 1)).flatten())
            
            # Sort H and find rank of H[0]
            rank = np.searchsorted(np.sort(H), H[0])
            
            # Check if H[0] is in the confidence region
            if rank < M - q:
                grid[i, j] = 1  # Parameter is in confidence region
            else:
                grid[i, j] = 0  # Parameter is NOT in confidence region

    return grid

def SPS_visualize(xx, yy, sig_st, E, search_sig_cr, search_tau, q, M, p1_opt=None, p2_opt=None):
    '''
    Visualizes SPS confidence regions and data
    
    Parameters:
    xx, yy: original data
    sig_st, E: parameters for unit conversion
    search_sig_cr: search range for sig_cr in absolute values
    search_tau: search range for tau in seconds
    q, M: SPS parameters
    p1_opt, p2_opt: optional optimal parameters from LSM to overlay
    '''
    
    # Convert to unit values
    xx_unit, yy_unit = prepare(xx, yy, sig_st, E)
    
    # Convert search ranges to unit parameters
    search_p1 = search_sig_cr / sig_st
    search_p2 = search_tau / tau0
    
    # Perform SPS analysis
    print("Running SPS analysis...")
    grid = SPS(xx_unit, yy_unit, search_p1, search_p2, q, M)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.scatter(xx, yy, color='red', label='Experimental data')
    
    # If optimal parameters are provided, plot the model curve
    if p1_opt is not None and p2_opt is not None:
        # Generate smooth curve for model
        x_curve = np.logspace(np.log10(min(xx_unit)), np.log10(max(xx_unit)), 100)
        y_curve = np.array([model(x, p1_opt, p2_opt) for x in x_curve])
        x_curve_abs, y_curve_abs = comeback(x_curve, y_curve, sig_st, E)
        
        ax1.plot(x_curve_abs, y_curve_abs, 'b-', linewidth=2, 
                label=f'Optimal model\nsig_cr={p1_opt*sig_st:.3f}, tau={p2_opt*1e-6:.2e}s')
    
    ax1.set_xlabel('Strain rate (1/s)')
    ax1.set_ylabel('Stress (Pa)')
    ax1.set_title('Experimental Data')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Right subplot: SPS confidence region contour plot
    P1, P2 = np.meshgrid(search_p1, search_p2)
    
    # Convert grid parameters to absolute values for contour labels
    P1_abs = P1 * sig_st
    P2_abs = P2 * tau0
    
    # Create contour plot for SPS confidence region
    contour = ax2.contourf(P1_abs, P2_abs, grid.T, levels=[0.5, 1.5], 
                          colors=['lightblue'], alpha=0.7)
    
    # Add contour lines for better visibility
    contour_lines = ax2.contour(P1_abs, P2_abs, grid.T, levels=[0.5], 
                               colors=['blue'])
    
    # If optimal parameters are provided, mark them
    if p1_opt is not None and p2_opt is not None:
        ax2.scatter(p1_opt * sig_st, p2_opt * 1e-6, color='red', s=150, 
                   marker='*', label='Optimal point', zorder=5)
    
    ax2.set_xlabel('sig_cr (Pa)')
    ax2.set_ylabel('tau (s)')
    ax2.set_title(f'SPS Confidence Region (q={q}, M={M})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Calculate and display confidence region statistics
    confidence_area = np.sum(grid)
    total_area = grid.size
    confidence_percentage = (confidence_area / total_area) * 100
    
    # Add text box with statistics
    textstr = f'Confidence region: {confidence_percentage:.1f}% of search space\n({confidence_area}/{total_area} points)'
    ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.8))

    
    print(f"SPS Analysis Results:")
    print(f"  Confidence region covers {confidence_percentage:.1f}% of search space")
    print(f"  {confidence_area} out of {total_area} parameter combinations are in the {100*(1-q/M):.1f}% confidence region")
    
    plt.tight_layout()
    plt.show()

    return fig, (ax1, ax2)

def LSM_visualize(xx, yy, sig_st, E, search_sig_cr, search_tau):
    '''
    Visualizes data, model fit, and LSM grid search results
    
    Parameters:
    xx, yy: original data
    sig_st, E: mechanical properties
    search_sig_cr: search range for sig_cr in absolute values (search_p1 * sig_st)
    search_tau: search range for tau in seconds (search_p2 * 1e-6)
    '''
    
    # Convert to unit values
    xx_unit, yy_unit = prepare(xx, yy, sig_st, E)
    
    # Convert search ranges to unit parameters
    search_p1 = search_sig_cr / sig_st
    search_p2 = search_tau / tau0
    
    # Perform LSM grid search
    opti_i, opti_j, summ_grid = LSM(xx_unit, yy_unit, search_p1, search_p2)
    p1_opt = search_p1[opti_i]
    p2_opt = search_p2[opti_j]
    
    # Convert optimal parameters back to absolute values
    sig_cr_opt = p1_opt * sig_st
    tau_opt = p2_opt * tau0
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Left subplot: Data and model curve
    # Generate smooth curve for model
    #x_curve = np.logspace(np.log10(min(xx_unit)), np.log10(max(xx_unit)), 100)
    x_curve = np.logspace(np.log10(1e-8), np.log10(10), 100)
    y_curve = np.array([model(x, p1_opt, p2_opt) for x in x_curve])
    
    # Convert back to absolute values for plotting
    x_curve_abs, y_curve_abs = comeback(x_curve, y_curve, sig_st, E)
    
    label_params = f'$\sigma_{{cr}}$={sig_cr_opt/1e6:.0f}$MPa$,\n$\\tau$={tau_opt*1e6:.0f}$\mu s$'

    ax1.scatter(xx, yy, color='red', label='Experimental data')
    ax1.plot(x_curve_abs, y_curve_abs, 'b-', label='Model LSM fit\n'+label_params)
    ax1.set_xlabel('Strain rate (1/s)')
    ax1.set_ylabel('Stress (MPa)')
    ax1.set_title('Data and Model Fit')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # Right subplot: Contour plot of LSM grid
    P1, P2 = np.meshgrid(search_p1, search_p2)
    
    # Convert grid parameters to absolute values for contour labels
    P1_abs = P1 * sig_st
    P2_abs = P2 * tau0
    
    contour = ax2.contour(P1_abs, P2_abs, summ_grid.T, levels=50, cmap='viridis')
    ax2.clabel(contour, inline=True, fontsize=6)
    ax2.scatter(sig_cr_opt, tau_opt, color='red', s=100, marker='*', 
                label=f'Optimal:\n'+label_params)
    ax2.set_xlabel('$\sigma_{cr}(MPa)$')
    ax2.set_ylabel('$\\tau(\mu s)$')
    ax2.set_title('LSM Grid Search')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Print optimal parameters
    print(f"Optimal parameters:")
    print(f"  sig_cr = {sig_cr_opt:.6f} Pa")
    print(f"  tau = {tau_opt:.6e} s")
    print(f"  Unit parameters: p1 = {p1_opt:.6f}, p2 = {p2_opt:.6f}")

    plt.tight_layout()
    plt.show()
    
    return fig, (ax1, ax2)

# demo of the module
if __name__ == "__main__":
    # Generate some test data
    true_sig_cr = 100e6  # 50 MPa
    true_tau = 50e-6     # 50 microsecond

    E = 200e9           # 200 GPa
    sig_st = 100e6      # 100 MPa

    # Generate data
    xx_unit, yy_unit = gen_data(true_sig_cr/sig_st, true_tau/tau0)
    xx_abs, yy_abs = comeback(xx_unit, yy_unit, sig_st, E)

    # Define search ranges
    search_sig_cr = np.linspace(0.5*true_sig_cr, 1.5*true_sig_cr, 50)  # 50-150 MPa
    search_tau = np.logspace(-6, -4, 100)         # 1-100 microseconds

    # Visualize LSM
    fig, axes = LSM_visualize(xx_abs, yy_abs, sig_st, E, search_sig_cr, search_tau)



# Archive of blood and tears: :)

# def model_LSM(data, search_p1, search_p2, true_p1, true_p2):
#     '''
#     LSM search in unit variable space with contour plot visualization
#     '''
#     xx, yy = data
    
#     # Calculate sum of squares grid
#     summ_grid = np.zeros((len(search_p1), len(search_p2)))
    
#     for i, p1 in enumerate(search_p1):
#         for j, p2 in enumerate(search_p2):
#             summ = 0
#             for k in range(len(xx)):
#                 y_model = model(xx[k], p1, p2)
#                 summ += (yy[k] - y_model)**2
#             summ_grid[i, j] = summ
    
#     # Find optimal parameters
#     opti_idx = np.unravel_index(np.argmin(summ_grid), summ_grid.shape)
#     opti_p1 = search_p1[opti_idx[0]]
#     opti_p2 = search_p2[opti_idx[1]]
#     opti_summ = summ_grid[opti_idx]
    
#     # Create plots
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
#     # Left plot: Model curve and generated dots
#     # Plot theoretical curves with both optimal and true parameters
#     x_curve = np.logspace(-8, 1, 1000)
    
#     # Optimal model curve
#     y_curve_opti = np.array([model(x, opti_p1, opti_p2) for x in x_curve])
    
#     # True model curve
#     y_curve_true = np.array([model(x, true_p1, true_p2) for x in x_curve])
    
#     ax1.loglog(x_curve, y_curve_opti, 'b-', linewidth=2, label=f'Optimal model: p1={opti_p1:.3f}, p2={opti_p2:.3f}')
#     ax1.loglog(x_curve, y_curve_true, 'g--', linewidth=2, alpha=0.7, label=f'True model: p1={true_p1:.1f}, p2={true_p2:.1f}')
#     ax1.scatter(xx, yy, color='red', s=50, zorder=5, label='Generated data')
    
#     # Mark transition points for both models
#     transition_x_opti = opti_p1 / opti_p2
#     transition_y_opti = model(transition_x_opti, opti_p1, opti_p2)
    
#     transition_x_true = true_p1 / true_p2
#     transition_y_true = model(transition_x_true, true_p1, true_p2)
    
#     ax1.axvline(transition_x_opti, color='blue', linestyle='--', alpha=0.7, label='Optimal transition')
#     ax1.axvline(transition_x_true, color='green', linestyle=':', alpha=0.7, label='True transition')
#     ax1.scatter([transition_x_opti], [transition_y_opti], color='blue', s=80, marker='s', zorder=6)
#     ax1.scatter([transition_x_true], [transition_y_true], color='green', s=70, marker='o', zorder=6)
    
#     ax1.set_xlabel('X (Dimensionless strain rate)')
#     ax1.set_ylabel('Y (Dimensionless stress)')
#     ax1.set_title('Model Curve and Data Points')
#     ax1.legend()
#     ax1.grid(True, alpha=0.3)
    
#     # Right plot: Contour plot of sum of squares IN LOGARITHM
#     P1, P2 = np.meshgrid(search_p1, search_p2, indexing='ij')
    
#     # Use LOG scale for sum of squares values
#     log_summ_grid = np.log10(summ_grid + 1e-15)  # Add small value to avoid log(0)
    
#     contour = ax2.contourf(P1, P2, log_summ_grid, levels=50, cmap='viridis', alpha=0.8)
    
#     # Add contour lines
#     contour_lines = ax2.contour(P1, P2, log_summ_grid, levels=10, colors='black', linewidths=0.5, alpha=0.7)
#     ax2.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f')
    
#     # Mark optimal point and true point
#     ax2.scatter(opti_p1, opti_p2, color='red', s=100, marker='s', 
#                label=f'Optimal point\np1={opti_p1:.3f}\np2={opti_p2:.3f}', zorder=5)
    
#     ax2.scatter(true_p1, true_p2, color='green', s=150, marker='.',
#                label=f'True point\np1={true_p1:.1f}\np2={true_p2:.1f}', zorder=5)
    
#     ax2.set_xlabel('p1')
#     ax2.set_ylabel('p2')
#     ax2.set_title('LSM Optimization Landscape (Log Contour)')
#     ax2.legend(bbox_to_anchor=(1.2, 1), loc='upper left')
#     ax2.grid(True, alpha=0.3)
    
#     # Add colorbar
#     plt.colorbar(contour, ax=ax2, label='log10(Sum of Squares)')
    
#     plt.tight_layout()
#     plt.show()
    
#     return opti_p1, opti_p2, summ_grid





# def sps_algorithm(data, search_p1, search_p2, q=50, M=500):
#     '''
#     SPS algo for model
#     '''
#     xx, yy = data
#     N = len(xx)

#     # SPS parameters
#     prob = 1 - q/M

#     # probability area - initialize grid properly
#     grid = np.zeros((len(search_p1), len(search_p2)))

#     for i, p1 in tqdm(enumerate(search_p1), total=len(search_p1)):
#         for j, p2 in enumerate(search_p2):
#             beta = np.random.choice([-1, 1], size=(M-1, N), p=[0.5, 0.5])
#             beta = np.vstack([np.ones((1, N)), beta])
#             delta = np.array([yy[k] - model(xx[k], p1, p2) for k in range(N)])
#             deriv = np.array([dmodeldp2(xx[k], p1, p2) for k in range(N)])
#             H = np.abs((beta @ (delta * deriv).reshape(-1, 1)).flatten())
            
#             # Sort H and find rank of H[0]
#             sorted_H = np.sort(H)  # Need to sort H first!
#             rank = np.searchsorted(sorted_H, H[0])
            
#             # Check if H[0] is in the confidence region
#             if rank < M - q:
#                 grid[i, j] = 1  # Parameter is in confidence region
#             else:
#                 grid[i, j] = 0  # Parameter is NOT in confidence region

#     return grid

# def plot_sps_with_data(data, grid, search_p1, search_p2, true_p1=None, true_p2=None):
#     '''
#     Plot SPS confidence region with test data on the left
#     '''
#     xx, yy_data = data
    
#     # Create figure with two subplots
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
#     # Left plot: Test data and model curves
#     x_curve = np.logspace(-8, 1, 1000)
    
#     # Plot true model curve if true parameters provided
#     if true_p1 is not None and true_p2 is not None:
#         y_true = np.array([model(x, true_p1, true_p2) for x in x_curve])
#         ax1.loglog(x_curve, y_true, 'g-', linewidth=2, label=f'True model: p1={true_p1:.2f}, p2={true_p2:.2f}')
        
#         # Mark true transition point
#         trans_true = true_p1 / true_p2
#         ax1.axvline(trans_true, color='green', linestyle='--', alpha=0.7, label='True transition')
    
#     # Plot data points
#     ax1.scatter(xx, yy_data, color='red', s=60, zorder=5, label='Test data points')
    
#     ax1.set_xlabel('X (Dimensionless strain rate)')
#     ax1.set_ylabel('Y (Dimensionless stress)')
#     ax1.set_title('Test Data and Model')
#     ax1.legend()
#     ax1.grid(True, alpha=0.3)
    
#     # Right plot: SPS confidence region
#     P1, P2 = np.meshgrid(search_p1, search_p2, indexing='ij')
    
#     # Plot the confidence region
#     contour = ax2.contourf(P1, P2, grid, levels=[-0.5, 0.5, 1.5], 
#                           colors=['white', 'blue'], alpha=0.7)
    
#     # Add contour lines
#     contour_lines = ax2.contour(P1, P2, grid, levels=[0.5], colors='darkblue', linewidths=2)
    
#     # Mark true parameters if provided
#     if true_p1 is not None and true_p2 is not None:
#         ax2.scatter(true_p1, true_p2, color='red', s=200, marker='*', 
#                    label=f'True parameters\np1={true_p1:.2f}, p2={true_p2:.2f}', 
#                    edgecolor='white', linewidth=2)
    
#     ax2.set_xlabel('p1')
#     ax2.set_ylabel('p2')
#     ax2.set_title('SPS Confidence Region\n(Blue = parameters in confidence set)')
#     ax2.legend()
#     ax2.grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.show()
    
#     # Print statistics
#     confidence_area = np.sum(grid) / grid.size * 100
#     print(f"Confidence region covers {confidence_area:.1f}% of parameter space")
#     print(f"Total parameters tested: {grid.size}")
#     print(f"Parameters in confidence region: {np.sum(grid)}")

# # Enhanced version with optimal parameters
# def plot_sps_with_data_enhanced(data, grid, search_p1, search_p2, true_p1=None, true_p2=None, optimal_p1=None, optimal_p2=None):
#     '''
#     Enhanced plot with test data, optimal parameters, and confidence region
#     '''
#     xx, yy_data = data
    
#     # Create figure with two subplots
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
#     # Left plot: Test data and model curves
#     x_curve = np.logspace(-8, 1, 1000)
    
#     # Plot true model curve
#     if true_p1 is not None and true_p2 is not None:
#         y_true = np.array([model(x, true_p1, true_p2) for x in x_curve])
#         ax1.loglog(x_curve, y_true, 'g-', linewidth=2, alpha=0.7, label=f'True model')
    
#     # Plot optimal model curve if available
#     if optimal_p1 is not None and optimal_p2 is not None:
#         y_optimal = np.array([model(x, optimal_p1, optimal_p2) for x in x_curve])
#         ax1.loglog(x_curve, y_optimal, 'b--', linewidth=2, label=f'Optimal fit')
    
#     # Plot data points
#     ax1.scatter(xx, yy_data, color='red', s=80, zorder=5, label=f'Test data (N={len(xx)})')
    
#     # Add transition points
#     if true_p1 is not None and true_p2 is not None:
#         trans_true = true_p1 / true_p2
#         ax1.axvline(trans_true, color='green', linestyle=':', alpha=0.7, label='True transition')
    
#     if optimal_p1 is not None and optimal_p2 is not None:
#         trans_optimal = optimal_p1 / optimal_p2
#         ax1.axvline(trans_optimal, color='blue', linestyle=':', alpha=0.7, label='Optimal transition')
    
#     ax1.set_xlabel('X (Dimensionless strain rate)')
#     ax1.set_ylabel('Y (Dimensionless stress)')
#     ax1.set_title('Test Data and Model Comparison')
#     ax1.legend()
#     ax1.grid(True, alpha=0.3)
    
#     # Right plot: SPS confidence region
#     P1, P2 = np.meshgrid(search_p1, search_p2, indexing='ij')
    
#     contour = ax2.contourf(P1, P2, grid, levels=[-0.5, 0.5, 1.5], 
#                           colors=['white', 'lightblue'], alpha=0.8)
    
#     contour_lines = ax2.contour(P1, P2, grid, levels=[0.5], colors='darkblue', linewidths=2)
    
#     # Mark parameters
#     if true_p1 is not None and true_p2 is not None:
#         ax2.scatter(true_p1, true_p2, color='green', s=250, marker='*', 
#                    label=f'True: p1={true_p1:.2f}, p2={true_p2:.2f}', 
#                    edgecolor='white', linewidth=2)
    
#     if optimal_p1 is not None and optimal_p2 is not None:
#         ax2.scatter(optimal_p1, optimal_p2, color='red', s=150, marker='D', 
#                    label=f'Optimal: p1={optimal_p1:.2f}, p2={optimal_p2:.2f}', 
#                    edgecolor='white', linewidth=2)
    
#     ax2.set_xlabel('p1')
#     ax2.set_ylabel('p2')
#     ax2.set_title(f'SPS Confidence Region (q={q}, M={M})\nBlue area = {np.sum(grid)/grid.size*100:.1f}% of parameter space')
#     ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     ax2.grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.show()

# # Complete working example
# if __name__ == "__main__":
#     # Your parameters
#     true_p1 = 1.0
#     true_p2 = 50.0
    
#     # Generate test data
#     data = gen_data(true_p1, true_p2)
    
#     # Search ranges
#     search_p1 = np.linspace(0.5, 1.5, 50)
#     search_p2 = np.linspace(30, 70, 50)
    
#     # SPS parameters
#     q = 50
#     M = 500
    
#     print("Running SPS algorithm...")
#     grid = sps_algorithm(data, search_p1, search_p2, q=q, M=M)
    
    
#     print("Plotting results...")
#     plot_sps_with_data_enhanced(data, grid, search_p1, search_p2, 
#                                true_p1, true_p2, true_p1, true_p2)


# Complete working example
# if __name__ == "__main__":
#     # Your true parameters and data
#     true_p1 = 1.0
#     true_p2 = 50.0
#     data = gen_data(true_p1, true_p2)
    
#     # Search ranges
#     search_p1 = np.linspace(0.5, 1.5, 100)
#     search_p2 = np.linspace(30, 70, 100)
    
#     # Run SPS algorithm
#     print("Running SPS algorithm...")
#     grid = sps_algorithm(data, search_p1, search_p2, q=10, M=100)
    
#     # Plot results
#     plot_sps_grid(grid, search_p1, search_p2, true_p1, true_p2)


# Example usage:
# if __name__ == "__main__":
#     # True parameters
#     true_p1 = 1.0
#     true_p2 = 50.0
    
#     # Generate test data
#     data = gen_data(true_p1, true_p2)

    

#     # Search ranges
#     search_p1 = np.linspace(0.8, 1.2, 100)
#     search_p2 = np.linspace(1, 100, 1000)
    
#     model_SPS(data, search_p1, search_p2, true_p1, true_p2)

    # # Run LSM with visualization
    # opti_p1, opti_p2, summ_grid = model_LSM(data, search_p1, search_p2, true_p1, true_p2)
    
    # print(f"True parameters: p1={true_p1}, p2={true_p2}")
    # print(f"Found parameters: p1={opti_p1:.3f}, p2={opti_p2:.3f}")
    # print(f"Minimum sum of squares: {np.min(summ_grid):.2e}")