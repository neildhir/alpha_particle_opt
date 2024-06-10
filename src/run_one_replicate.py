"""
Vanilla BO loop.
"""

from torch import Tensor, cat
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
import time

from tracing_example import f, build_train_data
from utils.bayes_opt import initialize_model, optimize_acqf_and_get_observation


def run(train_X: Tensor, train_Y: Tensor, iterations: int = 100, verbose: bool = True):

    mll, model, bounds = initialize_model(train_X, train_Y)

    for i in range(iterations):

        t0 = time.monotonic()

        # Re-fit model with new data: D =  D_old \cup D_new
        fit_gpytorch_model(mll)

        # Use best_f (expected energy retained) observed so far
        ei = ExpectedImprovement(model, best_f=train_Y.min(), maximize=False)

        # Optimise and get new observation
        new_x, new_f = optimize_acqf_and_get_observation(f, ei, bounds)

        # Update training points
        train_X = cat([train_X, new_x])
        train_Y = cat([train_Y, new_f])

        # Reinitialize the models so they are ready for fitting on next iteration
        mll, model = initialize_model(train_X, train_Y)  # TODO: with state-dict here?

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

    # TODO: add stoppping criterion
    # TODO: save and load data/model through state_dict once the model is trained
    # TODO: add B(x) constraints


if __name__ == "__main__":
    train_X, train_Y, bounds = build_train_data()
    run(train_X=train_X, train_Y=train_Y)
