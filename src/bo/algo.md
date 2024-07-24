# Pseudo-algorithm

- Set $\mathcal{D}_0 = \varnothing$
- Get features (Fourier coefficients) from tracing class: $(\mathbf{w}, \ldots) \leftarrow$ `TracerBoozer` and use in equation (1) of original paper to get corresponding target value (the _expected energy lost under this set of Fourier coefficients_ $\mathbf{w}$) giving the pair $(\mathbf{w}, y)$ as the first data point
- Let $\mathcal{D} \leftarrow \mathcal{D_0} \cup \{(\mathbf{w},y)\}$
- Cap number of iterations: $K$ (or some other stopping criterion e.g. convergence)
- While $i < K$:

    1. Fit surrogate model $f$ (multivariate Gaussian process with ARD scaled Matern-52 kernel) to training data $\mathcal{D}$
    2. Select new point $\mathbf{w}_{\text{new}}$ by optimizing acquisition function $\alpha(\mathbf{w}; \mathcal{D})$
        1. Optimise constrained acquisition function (we are using Expected Improvement as a baseline) where constraints are given by equation 13 and 14 of original paper
        2. The gradient based optimiser requires us to provide a generator of samples from the feasible set of $\mathbf{w}$ we use simple rejection sampling or e.g. NUTS/other MCMC method.
    3. Get target value $y_{\text{new}}$ by evaluating the tracing class with $\mathbf{w}_{\text{new}}$
    4. Update $\mathcal{D} \leftarrow \mathcal{D} \cup \{(\mathbf{w}_{\text{new}}, y_{\text{new}})\}$
    5. $i \leftarrow i + 1$
- End

## To-do

- [x] Use Fourier coefficients from all `nfp4` files and use these as the initial training set.
  - Assignee: @neil
- [x] Write a pseudo-sampler for feasible Fourier coefficients: just add some noise to the initial Fourier coefficients and treat these samples as if they are from VMEC (i.o.t. test the BO loop)
  - Assignee: @neil
- [ ] Update function arguments to reflect paper notation e.g. $\mathbf{w}$ instead of $\mathbf{x}$ in the objective function $\mathcal{J}(\cdot)$ as well as in constraint calc. (amongst other places)
  - Assignee: @misha
- [x] Box constraints
  - Assignee: @misha & @neil
- [ ] Change vmec output 'settings' (?) so that it doesn't generate any files each time it is being called
  - Assignee: @michael & @misha
- [x] Make class of the stuff in the `tracing_example.py` file
  - Assignee: @neil