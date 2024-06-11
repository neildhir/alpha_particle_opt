# Fast-ion Bayesian Optimization

Confining energetic ions such as alpha particles is a crucial consideration in the design of stellarators. While direct measurement of alpha confinement through numerical simulation of guiding-center trajectories has been deemed computationally expensive and noisy, proxy metrics – simplified measures of confinement – have been widely employed in the design process due to their computational tractability and proven effectiveness. However, the extent to which these proxies compromise the design optimality remains unclear when compared to relying on direct trajectory calculations. In this study, we employ Bayesian optimization to optimize stellarator designs for improved alpha particle confinement without resorting to proxy metrics. Specifically, we leverage Bayesian optimization to numerically optimize an objective function that measures alpha particle losses by simulating alpha particle trajectories. Despite the computational overhead associated with this approach, we demonstrate that Bayesian optimization can successfully generate configurations with low alpha particle losses, circumventing the need for proxy metrics and potentially yielding superior designs.

## Usage

Navigate to `src/` then run `python run_one_replicate.py` to run the optimisation.

## Misc

Used `Simsopt 0.7.4`
Set `OMP_NUM_THREADS=1` to ensure that MPI is not slowed down by OpenMP.
