# Pseudo-algorithm

- Set $\mathcal{D}_0 = \varnothing$
- Get features (Fourier coefficients) from tracing class: $(\mathbf{w}, \ldots) \leftarrow$ `TracerBoozer` and use in equation (1) of original paper to get corresponding target value (the _expected energy lost under this set of Fourier coefficients_ $\mathbf{w}$) giving the pair $(\mathbf{w}, y)$ as the first data point
- Let $\mathcal{D} \leftarrow \mathcal{D_0} \cup \{(\mathbf{w},y)\}$
- Cap number of iterations: $K$ (or some other stopping criterion e.g. convergence)
- While $i < K$:

    1. Fit surrogate model $f$ (multivariate Gaussian process with ARD scaled Matern-52 kernel) to training data $\mathcal{D}$
    2. Select new point $\mathbf{w}_{\text{new}}$ by optimizing acquisition function $\alpha(\mathbf{w}; \mathcal{D})$
        1. Optimise constrained acquisition function (we are using Expected Improvement as a baseline) where constraints are given by equation 13 and 14 of original paper
        2. The gradient based optimisier requires us to provide a generator of samples from the feasible set of $\mathbf{w}$ we use simple rejection sampling or e.g. NUTS/other MCMC method.
    3. Get target value $y_{\text{new}}$ by evaluating the tracing class with $\mathbf{w}_{\text{new}}$
    4. Update $\mathcal{D} \leftarrow \mathcal{D} \cup \{(\mathbf{w}_{\text{new}}, y_{\text{new}})\}$
    5. **QUESTION:** Presumably we need to re-rerun the tracing class `TracerBoozer` here, with the new Fourier coefficients, in order to get the updated version of the constraints, since $B(\mathbf{x}) \mid \mathbf{w}$.
    6. $i \leftarrow i + 1$
- End

## Questions

- What are the constraints a function of? $\mathbf{w}$ or $\mathbf{x}$?
- Is `x0` in `tracing_example.py` the same as $\mathbf{w}$? Is it the random initial position of a particle? If not, and it is initial Fourier coefficients, then why are they not the same as those in the input file?

## To-do

- [ ] Update function arguments to reflect paper notation e.g. $\mathbf{w}$ instead of $\mathbf{x}$ in the objective function $\mathcal{J}(\cdot)$ as well as in constraint calc. (amongst other places)
  - Assignee: @misha
- [ ] Change vmec output 'settings' (?) so that it doesn't generate any files each time it is being called
  - Assignee: @michael
- [ ] Make class of the stuff in the `tracing_example.py` file
  - Assignee: @neil