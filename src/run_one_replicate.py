"""
Vanilla BO loop.
"""

from torch import Tensor, cat
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.models.transforms import Normalize, Standardize
import math
import time

from tracing_example import f  # Objective function


def initialize_model(
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
    bounds = torch.stack([torch.zeros(d), torch.tensor([2 * math.pi])])
    input_transform = Normalize(d=d, bounds=bounds)

    # Surrogate model (gp)
    model = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        outcome_transform=Standardize(m=1),
        input_transform=input_transform,
    )  # Uses a scaled Matern kernel by default
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    # load state dict if it is passed
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model


def optimize_acqf_and_get_observation(acq_func: ExpectedImprovement) -> tuple[Tensor, Tensor]:
    """
    Optimizes the acquisition function and returns a new candidate and observation.

    Parameters
    ----------
    acq_func : ExpectedImprovement
        The acquisition function to be optimized.

    Returns
    -------
    tuple[Tensor, Tensor]
        New candidate and observation.
    """
    # optimize
    candidates, _ = optimize_acqf(acq_function=acq_func, bounds=bounds, num_restarts=100, raw_samples=10, q=1)
    # observe new values
    new_x = candidates.detach()  # Detach to avoid gradient updates
    new_obj = f(new_x)
    return new_x, new_obj


def run(train_X: Tensor, train_Y: Tensor, iterations: int = 100, verbose: bool = True):

    mll, model = initialize_model(train_X, train_Y)

    for i in range(iterations):

        t0 = time.monotonic()

        # fit the models
        fit_gpytorch_model(mll)

        # Use best_f observed so far
        ei = ExpectedImprovement(model, best_f=train_Y.max())  # TODO: check that this is not meant to be min

        # Optimise and get new observation
        new_x, new_f = optimize_acqf_and_get_observation(ei)

        # Update training points
        train_X = cat([train_X, new_x])
        train_Y = cat([train_Y, new_f])

        # Reinitialize the models so they are ready for fitting on next iteration
        mll, model = initialize_model(train_X, train_Y)

        t1 = time.monotonic()

        best_f = train_Y.min().item()

        if verbose:
            print(
                f"\nIteration {i}: best objective value = " f"{best_f}, " f"time = {t1-t0:>4.2f}.",
                end="",
            )
        else:
            print(".", end="")

    # TODO: add stoppping criterion


if __name__ == "__main__":
    train_X = None
    train_Y = None
    run(train_X=train_X, train_Y=train_Y)
