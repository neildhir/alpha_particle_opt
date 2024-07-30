#!/usr/bin/env python3
"""
This function basically generates points in the feasible polytope (bounds + non-linear constraints).

In the presence of nonlinear constraints sampling from the feasible set becomes a very hard problem to solve in a general fashion, so this is currently unsupported in the default initializer.

There are two options:

(i) pass as set of initial conditions to perform multi-start ascent from as batch_initial_conditions, or
(ii) write your own initializer callable with the same signature as gen_batch_initial_conditions that implements a strategy for generating such initial conditions and pass that in via a ic_generator kwarg to optimize_acqf.

In this work we will use the first option.
"""

from torch import rand, Tensor, from_numpy
from numpy import all as np_all
from numpy import vstack


def get_polytope_samples(n, bounds: Tensor, B_lower_constraint: callable, B_upper_constraint: callable) -> Tensor:

    valid_samples = []
    D = bounds.size(1)

    samples = rand(N, D)  # Generate random numbers between 0 and 1
    X = samples * (bounds[1] - bounds[0]) + bounds[0]  # Scale and shift the samples according to the bounds

    for x in X.numpy():
        if np_all(B_lower_constraint(x) <= 0) and np_all(B_upper_constraint(x) >= 0):
            valid_samples.append(x)  # Valid sample

    return from_numpy(vstack(valid_samples))


# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license:

# -----------------------------------------------------------------------------

# MIT License

# Copyright (c) Meta Platforms, Inc. and affiliates.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# -----------------------------------------------------------------------------

# Note that this file, ``initializers.py`` was modified from
# https://github.com/pytorch/botorch/blob/main/botorch/optim/initializers.py
# See https://github.com/pytorch/botorch/issues/1572
# for details.

"""
References

.. [Regis]
    R. G. Regis, C. A. Shoemaker. Combining radial basis function
    surrogates and dynamic coordinate search in high-dimensional
    expensive black-box optimization, Engineering Optimization, 2013.
"""

from __future__ import annotations

import warnings
from math import ceil
from typing import Dict, List, Optional, Tuple, Union

import torch
from botorch import settings
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.utils import is_nonnegative
from botorch.exceptions.errors import (
    BotorchTensorDimensionError,
    UnsupportedError,
)
from botorch.exceptions.warnings import (
    BadInitialCandidatesWarning,
    BotorchWarning,
    SamplingWarning,
)
from botorch.optim.utils import fix_features, get_X_baseline
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.sampling import (
    batched_multinomial,
    draw_sobol_samples,
    get_polytope_samples,
)
from botorch.utils.transforms import normalize, unnormalize
from torch import Tensor
from torch.distributions import Normal
from torch.quasirandom import SobolEngine


def gen_batch_initial_conditions_nonlinear(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int,
    num_restarts: int,
    raw_samples: int,
    fixed_features: Optional[Dict[int, float]] = None,
    options: Optional[Dict[str, Union[bool, float, int]]] = None,
    inequality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    equality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    nonlinear_constraint: Optional[callable] = None,
) -> Tensor:
    r"""Generate a batch of initial conditions for random-restart optimziation.

    TODO: Support t-batches of initial conditions.

    Args:
        acq_function: The acquisition function to be optimized.
        bounds: A `2 x d` tensor of lower and upper bounds for each column of `X`.
        q: The number of candidates to consider.
        num_restarts: The number of starting points for multistart acquisition
            function optimization.
        raw_samples: The number of raw samples to consider in the initialization
            heuristic. Note: if `sample_around_best` is True (the default is False),
            then `2 * raw_samples` samples are used.
        fixed_features: A map `{feature_index: value}` for features that
            should be fixed to a particular value during generation.
        options: Options for initial condition generation. For valid options see
            `initialize_q_batch` and `initialize_q_batch_nonneg`. If `options`
            contains a `nonnegative=True` entry, then `acq_function` is
            assumed to be non-negative (useful when using custom acquisition
            functions). In addition, an "init_batch_limit" option can be passed
            to specify the batch limit for the initialization. This is useful
            for avoiding memory limits when computing the batch posterior over
            raw samples.
        inequality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) >= rhs`.
        equality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) = rhs`.

    Returns:
        A `num_restarts x q x d` tensor of initial conditions.

    Example:
        >>> qEI = qExpectedImprovement(model, best_f=0.2)
        >>> bounds = torch.tensor([[0.], [1.]])
        >>> Xinit = gen_batch_initial_conditions(
        >>>     qEI, bounds, q=3, num_restarts=25, raw_samples=500
        >>> )
    """
    if bounds.isinf().any():
        raise NotImplementedError(
            "Currently only finite values in `bounds` are supported "
            "for generating initial conditions for optimization."
        )
    options = options or {}
    sample_around_best = options.get("sample_around_best", False)
    if sample_around_best and equality_constraints:
        raise UnsupportedError("Option 'sample_around_best' is not supported when equality" "constraints are present.")
    seed: Optional[int] = options.get("seed")
    batch_limit: Optional[int] = options.get("init_batch_limit", options.get("batch_limit"))
    factor, max_factor = 1, 5
    init_kwargs = {}
    device = bounds.device
    if "eta" in options:
        init_kwargs["eta"] = options.get("eta")
    if options.get("nonnegative") or is_nonnegative(acq_function):
        init_func = initialize_q_batch_nonneg
        if "alpha" in options:
            init_kwargs["alpha"] = options.get("alpha")
    else:
        init_func = initialize_q_batch

    q = 1 if q is None else q
    # the dimension the samples are drawn from
    effective_dim = bounds.shape[-1] * q
    if effective_dim > SobolEngine.MAXDIM and settings.debug.on():
        warnings.warn(
            f"Sample dimension q*d={effective_dim} exceeding Sobol max dimension "
            f"({SobolEngine.MAXDIM}). Using iid samples instead.",
            SamplingWarning,
        )

    while factor < max_factor:
        with warnings.catch_warnings(record=True) as ws:
            n = raw_samples * factor

            if sample_around_best:
                BotorchWarning(
                    "sample_around_best is True but there are "
                    "non-linear constraints provided. This might lead to "
                    "unexpected behavior since sampling points around "
                    "best has some random behavior which could violate "
                    "the non-linear constraints"
                )

            need = num_restarts * q
            all_points = torch.tensor([]).cpu()
            ndims = bounds.shape[1]
            counter = 0
            while all_points.shape[0] <= need:
                X_rnd = (
                    get_polytope_samples(
                        n=n * q,
                        bounds=bounds,
                        inequality_constraints=inequality_constraints,
                        equality_constraints=equality_constraints,
                        seed=seed,
                        n_burnin=options.get("n_burnin", 10000),
                        thinning=options.get("thinning", 32),
                    )
                    .view(n, q, -1)
                    .cpu()
                )
                X_rnd = X_rnd.reshape(-1, ndims)
                # TODO: modify to allow for multiple constraints
                where = torch.where(nonlinear_constraint(X_rnd) >= 0)[0]
                all_points = torch.cat([all_points, X_rnd[where, :]], axis=0)
                counter += 1
                if seed is not None:
                    seed += 1
                if counter >= 1000:
                    BotorchWarning(
                        "1000 iterations of trying to satisfy the "
                        "provided non-linear constraint. Recommend "
                        "revisiting the parameters of this constraint, "
                        "as it appears hard to satisfy it."
                    )
            X_rnd = all_points[:need, :].reshape(num_restarts, q, ndims)

            # sample points around best
            if sample_around_best:
                X_best_rnd = sample_points_around_best(
                    acq_function=acq_function,
                    n_discrete_points=n * q,
                    sigma=options.get("sample_around_best_sigma", 1e-3),
                    bounds=bounds,
                    subset_sigma=options.get("sample_around_best_subset_sigma", 1e-1),
                    prob_perturb=options.get("sample_around_best_prob_perturb"),
                )
                if X_best_rnd is not None:
                    X_rnd = torch.cat(
                        [
                            X_rnd,
                            X_best_rnd.view(n, q, bounds.shape[-1]).cpu(),
                        ],
                        dim=0,
                    )
            X_rnd = fix_features(X_rnd, fixed_features=fixed_features)
            with torch.no_grad():
                if batch_limit is None:
                    batch_limit = X_rnd.shape[0]
                Y_rnd_list = []
                start_idx = 0
                while start_idx < X_rnd.shape[0]:
                    end_idx = min(start_idx + batch_limit, X_rnd.shape[0])
                    Y_rnd_curr = acq_function(X_rnd[start_idx:end_idx].to(device=device)).cpu()
                    Y_rnd_list.append(Y_rnd_curr)
                    start_idx += batch_limit
                Y_rnd = torch.cat(Y_rnd_list)
            batch_initial_conditions = init_func(X=X_rnd, Y=Y_rnd, n=num_restarts, **init_kwargs).to(device=device)
            if not any(issubclass(w.category, BadInitialCandidatesWarning) for w in ws):
                return batch_initial_conditions
            if factor < max_factor:
                factor += 1
                if seed is not None:
                    seed += 1  # make sure to sample different X_rnd
    warnings.warn(
        "Unable to find non-zero acquisition function values - initial conditions " "are being selected randomly.",
        BadInitialCandidatesWarning,
    )
    return batch_initial_conditions


def initialize_q_batch(X: Tensor, Y: Tensor, n: int, eta: float = 1.0) -> Tensor:
    r"""Heuristic for selecting initial conditions for candidate generation.

    This heuristic selects points from `X` (without replacement) with probability
    proportional to `exp(eta * Z)`, where `Z = (Y - mean(Y)) / std(Y)` and `eta`
    is a temperature parameter.

    When using an acquisiton function that is non-negative and possibly zero
    over large areas of the feature space (e.g. qEI), you should use
    `initialize_q_batch_nonneg` instead.

    Args:
        X: A `b x batch_shape x q x d` tensor of `b` - `batch_shape` samples of
            `q`-batches from a d`-dim feature space. Typically, these are generated
            using qMC sampling.
        Y: A tensor of `b x batch_shape` outcomes associated with the samples.
            Typically, this is the value of the batch acquisition function to be
            maximized.
        n: The number of initial condition to be generated. Must be less than `b`.
        eta: Temperature parameter for weighting samples.

    Returns:
        A `n x batch_shape x q x d` tensor of `n` - `batch_shape` `q`-batch initial
        conditions, where each batch of `n x q x d` samples is selected independently.

    Example:
        >>> # To get `n=10` starting points of q-batch size `q=3`
        >>> # for model with `d=6`:
        >>> qUCB = qUpperConfidenceBound(model, beta=0.1)
        >>> Xrnd = torch.rand(500, 3, 6)
        >>> Xinit = initialize_q_batch(Xrnd, qUCB(Xrnd), 10)
    """
    n_samples = X.shape[0]
    batch_shape = X.shape[1:-2] or torch.Size()
    if n > n_samples:
        raise RuntimeError(f"n ({n}) cannot be larger than the number of " f"provided samples ({n_samples})")
    elif n == n_samples:
        return X

    Ystd = Y.std(dim=0)
    if torch.any(Ystd == 0):
        warnings.warn(
            "All acquisition values for raw samples points are the same for "
            "at least one batch. Choosing initial conditions at random.",
            BadInitialCandidatesWarning,
        )
        return X[torch.randperm(n=n_samples, device=X.device)][:n]

    max_val, max_idx = torch.max(Y, dim=0)
    Z = (Y - Y.mean(dim=0)) / Ystd
    etaZ = eta * Z
    weights = torch.exp(etaZ)
    while torch.isinf(weights).any():
        etaZ *= 0.5
        weights = torch.exp(etaZ)
    if batch_shape == torch.Size():
        idcs = torch.multinomial(weights, n)
    else:
        idcs = batched_multinomial(
            weights=weights.permute(*range(1, len(batch_shape) + 1), 0),
            num_samples=n,
        ).permute(-1, *range(len(batch_shape)))
    # make sure we get the maximum
    if max_idx not in idcs:
        idcs[-1] = max_idx
    if batch_shape == torch.Size():
        return X[idcs]
    else:
        return X.gather(dim=0, index=idcs.view(*idcs.shape, 1, 1).expand(n, *X.shape[1:]))


def initialize_q_batch_nonneg(X: Tensor, Y: Tensor, n: int, eta: float = 1.0, alpha: float = 1e-4) -> Tensor:
    r"""Heuristic for selecting initial conditions for non-neg. acquisition functions.

    This function is similar to `initialize_q_batch`, but designed specifically
    for acquisition functions that are non-negative and possibly zero over
    large areas of the feature space (e.g. qEI). All samples for which
    `Y < alpha * max(Y)` will be ignored (assuming that `Y` contains at least
    one positive value).

    Args:
        X: A `b x q x d` tensor of `b` samples of `q`-batches from a `d`-dim.
            feature space. Typically, these are generated using qMC.
        Y: A tensor of `b` outcomes associated with the samples. Typically, this
            is the value of the batch acquisition function to be maximized.
        n: The number of initial condition to be generated. Must be less than `b`.
        eta: Temperature parameter for weighting samples.
        alpha: The threshold (as a fraction of the maximum observed value) under
            which to ignore samples. All input samples for which
            `Y < alpha * max(Y)` will be ignored.

    Returns:
        A `n x q x d` tensor of `n` `q`-batch initial conditions.

    Example:
        >>> # To get `n=10` starting points of q-batch size `q=3`
        >>> # for model with `d=6`:
        >>> qEI = qExpectedImprovement(model, best_f=0.2)
        >>> Xrnd = torch.rand(500, 3, 6)
        >>> Xinit = initialize_q_batch(Xrnd, qEI(Xrnd), 10)
    """
    n_samples = X.shape[0]
    if n > n_samples:
        raise RuntimeError("n cannot be larger than the number of provided samples")
    elif n == n_samples:
        return X

    max_val, max_idx = torch.max(Y, dim=0)
    if torch.any(max_val <= 0):
        warnings.warn(
            "All acquisition values for raw sampled points are nonpositive, so "
            "initial conditions are being selected randomly.",
            BadInitialCandidatesWarning,
        )
        return X[torch.randperm(n=n_samples, device=X.device)][:n]

    # make sure there are at least `n` points with positive acquisition values
    pos = Y > 0
    num_pos = pos.sum().item()
    if num_pos < n:
        # select all positive points and then fill remaining quota with randomly
        # selected points
        remaining_indices = (~pos).nonzero(as_tuple=False).view(-1)
        rand_indices = torch.randperm(remaining_indices.shape[0], device=Y.device)
        sampled_remaining_indices = remaining_indices[rand_indices[: n - num_pos]]
        pos[sampled_remaining_indices] = 1
        return X[pos]
    # select points within alpha of max_val, iteratively decreasing alpha by a
    # factor of 10 as necessary
    alpha_pos = Y >= alpha * max_val
    while alpha_pos.sum() < n:
        alpha = 0.1 * alpha
        alpha_pos = Y >= alpha * max_val
    alpha_pos_idcs = torch.arange(len(Y), device=Y.device)[alpha_pos]
    weights = torch.exp(eta * (Y[alpha_pos] / max_val - 1))
    idcs = alpha_pos_idcs[torch.multinomial(weights, n)]
    if max_idx not in idcs:
        idcs[-1] = max_idx
    return X[idcs]


def sample_points_around_best(
    acq_function: AcquisitionFunction,
    n_discrete_points: int,
    sigma: float,
    bounds: Tensor,
    best_pct: float = 5.0,
    subset_sigma: float = 1e-1,
    prob_perturb: Optional[float] = None,
) -> Optional[Tensor]:
    r"""Find best points and sample nearby points.

    Args:
        acq_function: The acquisition function.
        n_discrete_points: The number of points to sample.
        sigma: The standard deviation of the additive gaussian noise for
            perturbing the best points.
        bounds: A `2 x d`-dim tensor containing the bounds.
        best_pct: The percentage of best points to perturb.
        subset_sigma: The standard deviation of the additive gaussian
            noise for perturbing a subset of dimensions of the best points.
        prob_perturb: The probability of perturbing each dimension.

    Returns:
        An optional `n_discrete_points x d`-dim tensor containing the
            sampled points. This is None if no baseline points are found.
    """
    X = get_X_baseline(acq_function=acq_function)
    if X is None:
        return
    with torch.no_grad():
        try:
            posterior = acq_function.model.posterior(X)
        except AttributeError:
            warnings.warn(
                "Failed to sample around previous best points.",
                BotorchWarning,
            )
            return
        mean = posterior.mean
        while mean.ndim > 2:
            # take average over batch dims
            mean = mean.mean(dim=0)
        try:
            f_pred = acq_function.objective(mean)
        # Some acquisition functions do not have an objective
        # and for some acquisition functions the objective is None
        except (AttributeError, TypeError):
            f_pred = mean
        if hasattr(acq_function, "maximize"):
            # make sure that the optimiztaion direction is set properly
            if not acq_function.maximize:
                f_pred = -f_pred
        try:
            # handle constraints for EHVI-based acquisition functions
            constraints = acq_function.constraints
            if constraints is not None:
                neg_violation = -torch.stack([c(mean).clamp_min(0.0) for c in constraints], dim=-1).sum(dim=-1)
                feas = neg_violation == 0
                if feas.any():
                    f_pred[~feas] = float("-inf")
                else:
                    # set objective equal to negative violation
                    f_pred = neg_violation
        except AttributeError:
            pass
        if f_pred.ndim == mean.ndim and f_pred.shape[-1] > 1:
            # multi-objective
            # find pareto set
            is_pareto = is_non_dominated(f_pred)
            best_X = X[is_pareto]
        else:
            if f_pred.shape[-1] == 1:
                f_pred = f_pred.squeeze(-1)
            n_best = max(1, round(X.shape[0] * best_pct / 100))
            # the view() is to ensure that best_idcs is not a scalar tensor
            best_idcs = torch.topk(f_pred, n_best).indices.view(-1)
            best_X = X[best_idcs]
    use_perturbed_sampling = best_X.shape[-1] >= 20 or prob_perturb is not None
    n_trunc_normal_points = n_discrete_points // 2 if use_perturbed_sampling else n_discrete_points
    perturbed_X = sample_truncated_normal_perturbations(
        X=best_X,
        n_discrete_points=n_trunc_normal_points,
        sigma=sigma,
        bounds=bounds,
    )
    if use_perturbed_sampling:
        perturbed_subset_dims_X = sample_perturbed_subset_dims(
            X=best_X,
            bounds=bounds,
            # ensure that we return n_discrete_points
            n_discrete_points=n_discrete_points - n_trunc_normal_points,
            sigma=sigma,
            prob_perturb=prob_perturb,
        )
        perturbed_X = torch.cat([perturbed_X, perturbed_subset_dims_X], dim=0)
        # shuffle points
        perm = torch.randperm(perturbed_X.shape[0], device=X.device)
        perturbed_X = perturbed_X[perm]
    return perturbed_X


def sample_truncated_normal_perturbations(
    X: Tensor,
    n_discrete_points: int,
    sigma: float,
    bounds: Tensor,
    qmc: bool = True,
) -> Tensor:
    r"""Sample points around `X`.

    Sample perturbed points around `X` such that the added perturbations
    are sampled from N(0, sigma^2 I) and truncated to be within [0,1]^d.

    Args:
        X: A `n x d`-dim tensor starting points.
        n_discrete_points: The number of points to sample.
        sigma: The standard deviation of the additive gaussian noise for
            perturbing the points.
        bounds: A `2 x d`-dim tensor containing the bounds.
        qmc: A boolean indicating whether to use qmc.

    Returns:
        A `n_discrete_points x d`-dim tensor containing the sampled points.
    """
    X = normalize(X, bounds=bounds)
    d = X.shape[1]
    # sample points from N(X_center, sigma^2 I), truncated to be within
    # [0, 1]^d.
    if X.shape[0] > 1:
        rand_indices = torch.randint(X.shape[0], (n_discrete_points,), device=X.device)
        X = X[rand_indices]
    if qmc:
        std_bounds = torch.zeros(2, d, dtype=X.dtype, device=X.device)
        std_bounds[1] = 1
        u = draw_sobol_samples(bounds=std_bounds, n=n_discrete_points, q=1).squeeze(1)
    else:
        u = torch.rand((n_discrete_points, d), dtype=X.dtype, device=X.device)
    # compute bounds to sample from
    a = -X
    b = 1 - X
    # compute z-score of bounds
    alpha = a / sigma
    beta = b / sigma
    normal = Normal(0, 1)
    cdf_alpha = normal.cdf(alpha)
    # use inverse transform
    perturbation = normal.icdf(cdf_alpha + u * (normal.cdf(beta) - cdf_alpha)) * sigma
    # add perturbation and clip points that are still outside
    perturbed_X = (X + perturbation).clamp(0.0, 1.0)
    return unnormalize(perturbed_X, bounds=bounds)


def sample_perturbed_subset_dims(
    X: Tensor,
    bounds: Tensor,
    n_discrete_points: int,
    sigma: float = 1e-1,
    qmc: bool = True,
    prob_perturb: Optional[float] = None,
) -> Tensor:
    r"""Sample around `X` by perturbing a subset of the dimensions.

    By default, dimensions are perturbed with probability equal to
    `min(20 / d, 1)`. As shown in [Regis]_, perturbing a small number
    of dimensions can be beneificial. The perturbations are sampled
    from N(0, sigma^2 I) and truncated to be within [0,1]^d.

    Args:
        X: A `n x d`-dim tensor starting points. `X`
            must be normalized to be within `[0, 1]^d`.
        bounds: The bounds to sample perturbed values from
        n_discrete_points: The number of points to sample.
        sigma: The standard deviation of the additive gaussian noise for
            perturbing the points.
        qmc: A boolean indicating whether to use qmc.
        prob_perturb: The probability of perturbing each dimension. If omitted,
            defaults to `min(20 / d, 1)`.

    Returns:
        A `n_discrete_points x d`-dim tensor containing the sampled points.

    """
    if bounds.ndim != 2:
        raise BotorchTensorDimensionError("bounds must be a `2 x d`-dim tensor.")
    elif X.ndim != 2:
        raise BotorchTensorDimensionError("X must be a `n x d`-dim tensor.")
    d = bounds.shape[-1]
    if prob_perturb is None:
        # Only perturb a subset of the features
        prob_perturb = min(20.0 / d, 1.0)

    if X.shape[0] == 1:
        X_cand = X.repeat(n_discrete_points, 1)
    else:
        rand_indices = torch.randint(X.shape[0], (n_discrete_points,), device=X.device)
        X_cand = X[rand_indices]
    pert = sample_truncated_normal_perturbations(
        X=X_cand,
        n_discrete_points=n_discrete_points,
        sigma=sigma,
        bounds=bounds,
        qmc=qmc,
    )

    # find cases where we are not perturbing any dimensions
    mask = (
        torch.rand(
            n_discrete_points,
            d,
            dtype=bounds.dtype,
            device=bounds.device,
        )
        <= prob_perturb
    )
    ind = (~mask).all(dim=-1).nonzero()
    # perturb `n_perturb` of the dimensions
    n_perturb = ceil(d * prob_perturb)
    perturb_mask = torch.zeros(d, dtype=mask.dtype, device=mask.device)
    perturb_mask[:n_perturb].fill_(1)
    # TODO: use batched `torch.randperm` when available:
    # https://github.com/pytorch/pytorch/issues/42502
    for idx in ind:
        mask[idx] = perturb_mask[torch.randperm(d, device=bounds.device)]
    # Create candidate points
    X_cand[mask] = pert[mask]
    return X_cand
