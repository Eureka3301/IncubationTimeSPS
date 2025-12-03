# IncubationTimeSPS
This is an implementation of SPS method for incubation time criterion model

## SPS algorithm for linear regression model
The algorithm used is taken from the article:

B.C. Csaji, M.C. Campi and E. Weyer.
*Sign-Perturbed Sums: A New System Identificaiton Approach for Contructing Exact Non-Asymptotic Confidence Regions in Linear Regression Models.*
IEEE Trans. on Signal Processing, 63, no.1: 169-181, 2015.

## Incubation Time model for linear loading

The IT model yield limit $\sigma_y$ for a given elastic stress $\sigma(t)=H(t)E\dot{\varepsilon}t$ looks like:


$$
\sigma_y = 
\begin{cases}
    \sigma_{cr} + \frac{\tau}{2} E \dot{\varepsilon} \quad  &   \dot{\varepsilon} < \frac{2 \sigma_{cr}}{E \tau} \\
    \sqrt{2 \sigma_{cr} E \tau \dot{\varepsilon}}    \quad  &   \dot{\varepsilon} > \frac{2 \sigma_{cr}}{E \tau}
\end{cases}
$$

One can use normalisation of this model to use unit variables as follows:

$$
Y = 
\begin{cases}
    p_1 + p_2 X         \quad  &   X < p_1/p_2 \\
    2 \sqrt{p_1 p_2 X}  \quad  &   X > p_1/p_2
\end{cases}
$$

Here:

$$ p_1 = \sigma_{cr} / \sigma_{st}  \quad \text{relation between critical stress parameter and one in static} $$
$$ p_2 = \tau / \tau_0              \quad \text{relation between incubation time and its order}(1\mu s) $$
$$ \varepsilon_{st} = \sigma_{st} / E               \quad \text{this is approximate strain at yielding} $$
$$ \dot{\varepsilon}_{st} = 2 \varepsilon_{st} / \tau_0 \quad \text{specific strain rate}$$
$$ Y = \sigma_y / \sigma_{st}       \quad \text{relation between yield stress and the one in static} $$
$$ X = \dot{\varepsilon}/\dot{\varepsilon}_{st} \quad \text{relation between strain rate and specific strain rate} $$

## Incubation Time model for pulse loading


The IT model yield limit $\sigma_y$ for a given elastic stress $\sigma(t)=H(t) \sigma_y$ looks like:


$$
\sigma_y = 
\begin{cases}
    \sigma_{cr} \dfrac{\tau}{t_y} \quad  &   t_y < \tau \\
    \sigma_{cr}                   \quad  &   t_y > \tau
\end{cases}
$$

One can use normalisation of this model to use unit variables as follows:

$$
Y = 
\begin{cases}
    p_1 \dfrac{p_2}{X}    \quad  &   X < p_2 \\
    p_1                   \quad  &   X > p_2
\end{cases}
$$

Here:

$$ p_1 = \sigma_{cr} / \sigma_{st}  \quad \text{relation between critical stress parameter and one in static} $$
$$ p_2 = \tau / \tau_0              \quad \text{relation between incubation time and its order}(1\mu s) $$
$$ Y = \sigma_y / \sigma_{st}       \quad \text{relation between yield stress and the one in static} $$
$$ X = t_y / \tau_0                 \quad \text{relation between time till fracture and its order} $$

## LSM algorithm implementation

On the given grid calculating residuals and summing their squares.
At the same time comparing residuals on the grid with relative error.

## SPS algorithm implementation

I want to write how it works for the model.
It is a bit harder than for linear regression.

## Grid refinement

Iterative remeshing centered on best LSM fit.
Mesh finess keeps the same, the search window is zoomed.

## Visuals

The left axes $\dot{\varepsilon}$ - $\sigma_y$.
The right axes $\sigma_y$ - $\tau$.

`visual_assembly`

2 types for the left:
dots and curves
3 types for the right:
dots, contours and areas

`prepare_LSM_SPS`
calculating for assembly

## Optimisation

adding norming by np.linalg.eigh (for symmetrical matrix)

(tau_max - tau_min ) / tau_min --> search optimisation

upload/download from file mb some GUI
