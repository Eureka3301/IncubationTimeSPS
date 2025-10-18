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
    p_1 + p_2 X         \qed     X \geq p_2 \\
    2*\sqrt{p_1 p_2 X}  \qed     X \leq p_2
\end{cases}
$$

Here:
$$
\begin{itemize}
    \item
    p_1 = \sigma_{st} / \sigma_st \qed \text{relation between critical stress parameter and one in static}
    \item
    p_2 = \tau / \tau_0 \qed \text{relation between incubation time and its order}
    \item
    Y = \sigma_y / \sigma_{st} \qed \text{relation between yield stress and one in static}
    \item
    X = \dot{\varepsilon} / \varepsilon_{st}/\tau_0 \qed \text{relation between strain rate and specific strain rate}
    \item
    \varepsilon_{st} = \sigma_{st} / E \qed \text{this is approximately strain at yielding}
\end{itemize}
$$
