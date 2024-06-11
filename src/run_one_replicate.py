"""
Vanilla BO loop.
"""

from torch import Tensor, vstack
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement
import time

from tracing_example import f, build_train_data
from utils.bayes_opt import build_surrogate_model, optimize_acqf_and_get_observation


def run(train_X: Tensor, train_Y: Tensor, bounds: Tensor, iterations: int = 5, verbose: bool = True) -> None:
    """
    Run a full Bayesian optimization loop.

    Parameters
    ----------
    train_X : Tensor
        Input features (training data, Fourier coefficients)
    train_Y : Tensor
        Target (expected energy loss)
    bounds : Tensor
        Bounds of the optimization space
    iterations : int, optional
        Number of operations, by default 5 -- will be replaced with a stopping criterion
    verbose : bool, optional
        Print stuff or not, by default True
    """

    mll, model = build_surrogate_model(train_X, train_Y)
    for i in range(iterations):

        t0 = time.monotonic()

        # Re-fit model with new data: D =  D_old \cup D_new
        fit_gpytorch_mll(mll)

        # Use best_f (expected energy loss) observed so far
        ei = ExpectedImprovement(model, best_f=train_Y.min(), maximize=False)

        # Optimise and get new observation
        # TODO: the bounds here are currently placeholders until we include correct inequality constraints, then we can remove bounds or replace upper and lower with -inf and inf respectively
        # TODO: add stoppping criterion
        new_x, new_f = optimize_acqf_and_get_observation(f, ei, bounds)

        # Update training points
        train_X = vstack([train_X, new_x])
        train_Y = vstack([train_Y, new_f])

        # Reinitialize the models so they are ready for fitting on next iteration
        mll, model = build_surrogate_model(train_X, train_Y)  # TODO: with state-dict here?

        t1 = time.monotonic()
        best_f = train_Y.min().item()
        if verbose:
            print(
                f"\nIteration {i}: best objective value = " f"{best_f:>4.2f}, " f"time = {t1-t0:>4.2f}.",
                end="",
            )
        else:
            print(".", end="")

        # Stopping criterion

    # TODO: save and load data/model through state_dict once the model is trained
    # TODO: add B(x) constraints
    # TODO: save results module


# Reasonable domains for the Fourier coefficients?
train_X, train_Y, bounds = build_train_data()
run(train_X=train_X, train_Y=train_Y, bounds=bounds)

if __name__ == "__main__":
    train_X, train_Y, bounds = build_train_data()
    run(train_X=train_X, train_Y=train_Y, bounds=bounds)
