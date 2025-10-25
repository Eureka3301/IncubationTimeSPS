import numpy as np
from mechanics import *
from optimizers import *

import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science','no-latex'])

def LSM_visualize(xx, yy, sig_st, E, search_sig_cr, search_tau):
    '''
    Visualizes data, SPS grid search results
    
    Parameters:
    xx, yy: original data
    sig_st, E: mechanical properties
    search_sig_cr: search range for sig_cr in absolute values (search_p1 * sig_st)
    search_tau: search range for tau in seconds (search_p2 * 1e-6)
    '''
    
    # Convert to unit values
    xx_unit, yy_unit = comb(xx, yy, sig_st, E)
    
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
    x_curve_abs, y_curve_abs = ruffle(x_curve, y_curve, sig_st, E)
    
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


def SPS_visualize(xx, yy, sig_st, E, search_sig_cr, search_tau, q, M):
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
    xx_unit, yy_unit = comb(xx, yy, sig_st, E)
    
    # Convert search ranges to unit parameters
    search_p1 = search_sig_cr / sig_st
    search_p2 = search_tau / tau0
    
    # Perform SPS analysis
    print("Running SPS analysis...")
    SPS_grid = SPS(xx_unit, yy_unit, search_p1, search_p2, q, M)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.scatter(xx, yy, color='red', label='Experimental data')
    
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
    contour = ax2.contourf(P1_abs, P2_abs, SPS_grid.T, levels=[0.5, 1.5], 
                          colors=['lightblue'], alpha=0.7)
    
    # Add contour lines for better visibility
    contour_lines = ax2.contour(P1_abs, P2_abs, SPS_grid.T, levels=[0.5], 
                               colors=['blue'])
    
    ax2.set_xlabel('sig_cr (Pa)')
    ax2.set_ylabel('tau (s)')
    ax2.set_title(f'SPS Confidence Region (q={q}, M={M})')
    ax2.grid(True, alpha=0.3)
    
    # Calculate and display confidence region statistics
    confidence_area = np.sum(SPS_grid)
    total_area = SPS_grid.size
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


def LSM_SPS_visualize(xx, yy, sig_st, E, search_sig_cr, search_tau, q, M, refinement_depth = 1, refinement_scale = 0.1):
    '''
    Visualizes data, model fit, LSM and SPS grid search results
    
    Parameters:
    xx, yy: original data
    sig_st, E: mechanical properties
    search_sig_cr: search range for sig_cr in absolute values (search_p1 * sig_st)
    search_tau: search range for tau in seconds (search_p2 * 1e-6)
    q, M: SPS parameters
    refinement_depth: grid remeshing iteration quantity
    refinement_scale: grid bounds scaling factor
    '''
    
    # Convert to unit values
    xx_unit, yy_unit = comb(xx, yy, sig_st, E)
    
    # Convert search ranges to unit parameters
    search_p1 = search_sig_cr / sig_st
    search_p2 = search_tau / tau0
    
    print(f"Running 1 LSM analysis...")
    opti_i, opti_j, summ_grid = LSM(xx_unit, yy_unit, search_p1, search_p2)

    # Perform deep LSM grid search
    for k in range(1, refinement_depth):
        p1_opt = search_p1[opti_i]
        p2_opt = search_p2[opti_j]
        print(f'Unit parameters {k} step:')
        print(f"  p1 = {p1_opt:.6f}")
        print(f"  p2 = {p2_opt:.6f}")

        search_p1, search_p2 = refine_grid(opti_i, opti_j, search_p1, search_p2)

        print(f"Running {k+1} LSM analysis...")
        opti_i, opti_j, summ_grid = LSM(xx_unit, yy_unit, search_p1, search_p2)
    
    p1_opt = search_p1[opti_i]
    p2_opt = search_p2[opti_j]

    print(f'Unit parameters {refinement_depth} step:')
    print(f"  p1 = {p1_opt:.6f}")
    print(f"  p2 = {p2_opt:.6f}")

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
    x_curve_abs, y_curve_abs = ruffle(x_curve, y_curve, sig_st, E)
    
    label_params = f'$\sigma_{{cr}}$={sig_cr_opt/1e6:.0f}$MPa$,\n$\\tau$={tau_opt:.1e}$ s$'

    ax1.scatter(xx, yy, color='red', label='Experimental data')
    ax1.plot(x_curve_abs, y_curve_abs, 'b-', label='Model LSM fit\n'+label_params)
    ax1.set_xlabel('Strain rate (1/s)')
    ax1.set_ylabel('Stress (MPa)')
    ax1.set_title('Data and Model Fit')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # Perform SPS analysis
    print("Running SPS analysis...")
    SPS_grid = SPS(xx_unit, yy_unit, search_p1, search_p2, q, M)

    # Right subplot: SPS confidence region contour plot
    P1, P2 = np.meshgrid(search_p1, search_p2)
    
    # Convert grid parameters to absolute values for contour labels
    P1_abs = P1 * sig_st
    P2_abs = P2 * tau0
    
    # Create contour plot for SPS confidence region
    contour = ax2.contourf(P1_abs, P2_abs, SPS_grid.T, levels=[0.5, 1.5], 
                          colors=['lightblue'], alpha=0.7)
    
    # Add contour lines for better visibility
    contour_lines = ax2.contour(P1_abs, P2_abs, SPS_grid.T, levels=[0.5], 
                               colors=['blue'])
    
    ax2.set_title(f'SPS Confidence Region (q={q}, M={M})')
    
    # Calculate and display confidence region statistics
    confidence_area = np.sum(SPS_grid)
    total_area = SPS_grid.size
    confidence_percentage = (confidence_area / total_area) * 100
    
    # Add text box with statistics
    textstr = f'Confidence region: {confidence_percentage:.1f}% of search space\n({confidence_area:.0f}/{total_area:.0f} points)'
    ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.8))

    print(f"SPS Analysis Results:")
    print(f"  Confidence region covers {confidence_percentage:.1f}% of search space")
    print(f"  {confidence_area} out of {total_area} parameter combinations are in the {100*(1-q/M):.1f}% confidence region")
    
    # Right subplot: Contour plot of LSM grid
    P1, P2 = np.meshgrid(search_p1, search_p2)
    
    # Convert grid parameters to absolute values for contour labels
    P1_abs = P1 * sig_st
    P2_abs = P2 * tau0
    
    contour = ax2.contour(P1_abs, P2_abs, summ_grid.T, levels=50, cmap='viridis')
    ax2.clabel(contour, inline=True, fontsize=6)
    ax2.scatter(sig_cr_opt, tau_opt, color='red', s=50, marker='*', 
                label=f'Optimal:\n'+label_params)
    ax2.set_xlabel('$\\sigma_{cr}(MPa)$')
    ax2.set_ylabel('$\\tau(\mu s)$')
    ax2.legend(fancybox=True,
               frameon=True,
               loc='best')
    ax2.grid(True, alpha=0.3)
    
    # Print optimal parameters
    print(f"Unit parameters:")
    print(f"  p1 = {p1_opt:.6f}")
    print(f"  p2 = {p2_opt:.6f}")
    print(f"Optimal parameters:")
    print(f"  sig_cr = {sig_cr_opt/1e6:.0f} Pa")
    print(f"  tau = {tau_opt:.1e} s")

    plt.tight_layout()
    plt.show()
    
    return fig, (ax1, ax2)
