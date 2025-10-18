# IncubationTimeSPS
This is an implementation of SPS method for incubation time criterion model

## SPS algorithm for linear regression model
The algorithm used is taken from the article:

B.C. Csaji, M.C. Campi and E. Weyer.
*Sign-Perturbed Sums: A New System Identificaiton Approach for Contructing Exact Non-Asymptotic Confidence Regions in Linear Regression Models.*
IEEE Trans. on Signal Processing, 63, no.1: 169-181, 2015.

## Incubation time model

$$
Y = 
\begin{cases}
    p_1 + p_2 X         \quad  &   X \leq p_2 \\
    2*\sqrt{p_1 p_2 X}  \quad  &   X \geq p_2
\end{cases}
$$

Here:

$$ p_1 = \sigma_{st} / \sigma_st \quad \text{relation between critical stress parameter and one in static} $$
$$ p_2 = \tau / \tau_0 \quad \text{relation between incubation time and its order} $$
$$ Y = \sigma_y / \sigma_{st} \quad \text{relation between yield stress and one in static} $$
$$ X = \dot{\varepsilon} / \varepsilon_{st}/\tau_0 \quad \text{relation between strain rate and specific strain rate} $$
$$ \varepsilon_{st} = \sigma_{st} / E \quad \text{this is approximate strain at yielding} $$
