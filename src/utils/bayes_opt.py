from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from torch import Tensor, tensor

from src.tracing_example import acqf_nonlinear_inequality_constraints


def build_surrogate_model(
    train_X: Tensor, train_Y: Tensor, state_dict=None
) -> tuple[ExactMarginalLogLikelihood, SingleTaskGP]:
    """
    Initialize the Krigeing model (Gaussian process regression) for the BO loop.

    Parameters
    ----------
    train_X : Tensor
        Training data.
    train_Y : Tensor
        Training labels.
    state_dict : _type_, optional
        _description_, by default None

    Returns
    -------
    tuple[ExactMarginalLogLikelihood, SingleTaskGP]
        The model and the marginal log likelihood.
    """

    # Build surrogate model (gp)
    model = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
    )  # Uses a scaled Matern kernel by default
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    # load state dict if it is passed
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model


def optimize_acqf_and_get_observation(
    f: callable, acq_func: ExpectedImprovement, bounds: Tensor
) -> tuple[Tensor, Tensor]:
    """
    Optimizes the acquisition function and returns a new candidate and observation.

    Parameters
    ----------
    f : callable
        The objective function to be optimized.
    acq_func : ExpectedImprovement
        The acquisition function to be optimized.
    bounds : Tensor
        The bounds of the optimization space.

    Returns
    -------
    tuple[Tensor, Tensor]
        New candidate and observation.
    """

    constraints = acqf_nonlinear_inequality_constraints
    stopping_criterion = None  # Placeholder
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,  # TODO: remove bounds once have finished inequality constraints or set to -inf and inf
        nonlinear_inequality_constraints=constraints,
        num_restarts=10,  # XXX: perhaps reduce a spot
        raw_samples=512,  # XXX: perhaps reduce a spot
        q=1,  # Explore methods which allow q > 1
    )
    # observe new values
    new_x = candidates.detach()  # Detach to avoid gradient updates
    raise ValueError(
        "We should scale this to the unit cube but transform it back to the correct space before evaluating the objective function."
    )
    new_obj = f(new_x.numpy().flatten())  # This is cumbersome, re-write
    return new_x, new_obj
