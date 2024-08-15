from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from torch import Tensor
from typing import List, Tuple, Callable, Union


def build_surrogate_model(
    train_X: Tensor, train_Y: Tensor, bounds: Tensor, state_dict=None
) -> Tuple[ExactMarginalLogLikelihood, SingleTaskGP]:
    """
    Initialize the Krigeing model (Gaussian process regression) for the BO loop.

    Parameters
    ----------
    train_X : Tensor
        Training data.
    train_Y : Tensor
        Training labels.
    bounds : Tensor
        The bounds of the optimization space.
    state_dict : _type_, optional
        _description_, by default None

    Returns
    -------
    tuple[ExactMarginalLogLikelihood, SingleTaskGP]
        The model and the marginal log likelihood.
    """

    # Build surrogate model, kernel uses ARD by default
    model = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        outcome_transform=Standardize(m=1),
        input_transform=Normalize(d=train_X.size(1), bounds=bounds),
    )

    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    # Load state dict if it is passed
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model


def optimize_acqf_and_get_new_point(
    f: callable,
    ic_generator: callable,
    acq_func: ExpectedImprovement,
    bounds: Tensor,
    constraint: List[Tuple[Callable, bool]],
    SMOKE_TEST: bool,
) -> Tuple[Tensor, Tensor]:
    """
    Optimizes the acquisition function and returns a new candidate and observation.

    Parameters
    ----------
    f : callable
        The objective function to be optimized.
    acq_func : ExpectedImprovement
        The acquisition function to be optimized.
    bounds : Tensor
        A `2 x d` tensor of lower and upper bounds for each column of `X` (if inequality_constraints is provided, these bounds can be -inf and +inf, respectively).
    constraints : list[tuple[callable, bool]]
        Magnetic field strength constraints as a function of x, see eq. (13) and (14) of [1], section 4.2.

    Returns
    -------
    tuple[Tensor, Tensor]
        New candidate and observation.

    Notes
    -----
    The ic_generator is a function that generates initial conditions (starting points) for the optimization that satisfy the nonlinear constraints. It is crucial because it helps the optimizer start from feasible points.

    References
    ----------
    [1] Bindel, David, Matt Landreman, and Misha Padidar. "Direct optimization of fast-ion confinement in stellarators." Plasma Physics and Controlled Fusion 65.6 (2023): 065012.
    """

    NUM_RESTARTS = 4 if SMOKE_TEST else 8
    RAW_SAMPLES = 32 if SMOKE_TEST else 128

    # Initial condition (IC) generation is a fairly unsupported feature in BoTorch (at the time or writing). For details see: https://github.com/pytorch/botorch/issues/1572

    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=1,  # Explore methods which allow q > 1
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
        nonlinear_inequality_constraints=constraint,
        ic_generator=ic_generator,
    )  # Should also explore fixed features (i.e. fixed Fourier coefficients)

    # Observe new values
    new_x = candidates.detach()  # Detach to avoid gradient updates
    new_obj = f(new_x.numpy().flatten())  # This is cumbersome, re-write

    return new_x, new_obj
