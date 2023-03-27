# NaN Warning

## Description

Error message:
**solver_ipopt_ParallelMultistartNlp0_V:nlp_jac_g failed: NaN detected for output jac_g_x, at nonzero index 243 (row 61, col 50)**

* index of variable (x) = 50, i.e., $s_{50}$, the density of L3 at time 7
* index of constraint (g) = 61, i.e., the dynamics constraint of speed at L3 at time 7, i.e., $s_{53}$
* occurs at timestep 0, when solving for the state value.

## Constraint Expression

$$
\begin{align*}
    g_{61}(x) &= s_{53} \\
    &+ 0.555556 \cdot \left(v_{free} \exp{\left(-\frac{1}{a} \left(\frac{s_{50}}{\rho_{crit}}\right)^{a}\right)} - s_{53}\right) \\
    &+ 0.00277778 s_{53} \cdot \left(s_{52} - s_{53}\right) \\
    &- 33.3333 \cdot \frac{\max\left(\min\left(s_{50}, \rho_{crit}\right), d_{20}\right) - s_{50}}{s_{50} + 40} \\
    &- 1.694445 \cdot 10^{-5} \cdot \frac{a_{1} s_{53}}{s_{50} + 40} \\
\end{align*}
$$

## Constraint Jacobian Expression

$$
\begin{align*}
    \frac{dg_{61}(x)}{ds_{50}} &= (g_{61}(x) \ge 0) \cdot \Bigg( \\
    &- 0.555556 \cdot \left( \frac{v_{free}}{\rho_{crit}} \exp{\left(-\frac{1}{a} \left(\frac{s_{50}}{\rho_{crit}}\right)^a\right)} \left(\frac{s_{50}}{\rho_{crit}}\right)^{a - 1} \right) \\
    &- \frac{33.3333}{s_{50} + 40} \cdot \left(
        ((d_{20} <= \min{(s_{50}, \rho_{crit})}) \cdot (s_{50} \le \rho_{crit}) - 1)
        - \frac{\max{\left(\min{(s_{50}, \rho_{crit})}, d_{20}\right)} - s_{50}}{s_{50} + 40}
        \right) \\
    &+ \frac{3.38889 \cdot 10^{-5} a_{1} s_{53}}{(s_{50} + 40)^2} \\
    &\Bigg) \\
\end{align*}
$$
