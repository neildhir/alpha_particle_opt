from os import environ

from torch import Tensor, vstack
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement
import time

from tracing_example import f, get_init_BO_params
from bo.bayes_opt import build_surrogate_model, optimize_acqf_and_get_new_point

SMOKE_TEST = environ.get("SMOKE_TEST")  # TODO: finish this


# XXX: in this version of the method, the constraints are updated based on the new Fourier coefficients found during the optimisation
def run_with_constraints_update(
    train_X: Tensor,
    train_Y: Tensor,
    bounds: Tensor,
    nonlinear_inequality_constraints: list[tuple[callable, bool]],
    SMOKE_TEST: bool,
    iterations: int = 5,
    verbose: bool = True,
) -> None:
    """
    Runs a Bayesian optimization loop up to stopping criterion or until number of iterations have been exhausted.

    This version of the method updates the constraints based on the new Fourier coefficients found during the optimisation.

    Parameters
    ----------
    train_X : Tensor
        Input features (training data, Fourier coefficients)
    train_Y : Tensor
        Target (expected energy loss for this shape of the plasma boundary represented by the Fourier coefficients)
    bounds : Tensor
        Bounds of the optimization space
    nonlinear_inequality_constraints : list[tuple[callable, bool]]
        Magnetic field strength constraints as a function of x
    iterations : int, optional
        Number of operations, by default 5 -- will be replaced with a stopping criterion
    verbose : bool, optional
        Print stuff or not, by default True
    """

    iterations = iterations if not SMOKE_TEST else 4
    for i in range(iterations):

        t0 = time.monotonic()

        # Uses the available data to build the surrogate model
        mll, model = build_surrogate_model(train_X, train_Y)  # TODO: with state-dict here?
        # Fit model with data: D =  D_old \cup D_new
        fit_gpytorch_mll(mll)

        # Use best_f (expected energy loss) observed so far
        ei = ExpectedImprovement(model, best_f=train_Y.min(), maximize=False)

        # Optimise and get new observation
        # TODO: add stoppping criterion
        new_x, new_f = optimize_acqf_and_get_new_point(f, ei, bounds, nonlinear_inequality_constraints, SMOKE_TEST)

        # Update training points
        train_X = vstack([train_X, new_x])
        train_Y = vstack([train_Y, new_f])

        # Update nonlinear constraints by updating the B field values by updating the tracing class with the new Fourier coefficients: new_x
        update_magnetic_field_constraints(
            train_X
        )  # TODO: we should probably be using the best Fourier coefficients found so far, not necessarily the most recent ones
        # TODO: really this should be a call, something like Tracer.update(new_x), which updates the internal state of the tracer class with the new Fourier coefficients

        t1 = time.monotonic()

        best_f = train_Y.min().item()
        if verbose:
            print(
                f"\nIteration {i}: best objective value = " f"{best_f:>4.2f}, " f"time = {t1-t0:>4.2f}.",
                end="",
            )
        else:
            print(".", end="")

    # TODO: save and load data/model through state_dict once the model is trained
    # TODO: save results module


def run(
    train_X: Tensor,
    train_Y: Tensor,
    bounds: Tensor,
    nonlinear_inequality_constraints: list[tuple[callable, bool]],
    SMOKE_TEST: bool,
    iterations: int = 5,
    verbose: bool = True,
) -> None:
    """
    Runs a Bayesian optimization loop up to stopping criterion or until number of iterations have been exhausted.

    Parameters
    ----------
    train_X : Tensor
        Input features (training data, Fourier coefficients)
    train_Y : Tensor
        Target (expected energy loss)
    bounds : Tensor
        Bounds of the optimization space
    nonlinear_inequality_constraints : list[tuple[callable, bool]]
        Magnetic field strength constraints as a function of x
    iterations : int, optional
        Number of operations, by default 5 -- will be replaced with a stopping criterion
    verbose : bool, optional
        Print stuff or not, by default True
    """

    mll, model = build_surrogate_model(train_X, train_Y)

    iterations = iterations if not SMOKE_TEST else 4
    for i in range(iterations):

        t0 = time.monotonic()

        # Re-fit model with new data: D =  D_old \cup D_new
        fit_gpytorch_mll(mll)

        # Use best_f (expected energy loss) observed so far
        ei = ExpectedImprovement(model, best_f=train_Y.min(), maximize=False)

        # Optimise and get new observation
        # TODO: add stoppping criterion
        new_x, new_f = optimize_acqf_and_get_new_point(f, ei, bounds, nonlinear_inequality_constraints, SMOKE_TEST)

        # Update training points
        train_X = vstack([train_X, new_x])
        train_Y = vstack([train_Y, new_f])

        # Append to the dataset and reinitialize the model so that it is ready for fitting on next iteration
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

    # TODO: save and load data/model through state_dict once the model is trained
    # TODO: save results module


train_X, train_Y, bounds, constraints = get_init_BO_params()
SMOKE_TEST = True
run(train_X, train_Y, bounds, constraints, SMOKE_TEST)

if __name__ == "__main__":
    train_X, train_Y, bounds, constraints = get_init_BO_params()
    SMOKE_TEST = True
    run(train_X, train_Y, bounds, constraints, SMOKE_TEST)
