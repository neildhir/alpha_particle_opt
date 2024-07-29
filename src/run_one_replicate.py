from os import environ

from torch import Tensor, vstack
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement
import time

from tracing_example import StellaratorDesign
from bo.bayes_opt import build_surrogate_model, optimize_acqf_and_get_new_point
import os
from torch import load

SMOKE_TEST = environ.get("SMOKE_TEST")  # TODO: finish this


def run(
    f: callable,
    ic_generator: callable,
    train_X: Tensor,
    train_Y: Tensor,
    bounds: Tensor,
    nonlinear_inequality_constraints: list[tuple[callable, bool]],
    SMOKE_TEST: bool = False,
    iterations: int = 50,
    verbose: bool = True,
) -> None:
    """
    Runs a Bayesian optimization loop up to stopping criterion or until number of iterations have been exhausted.

    Parameters
    ----------
    f : callable,
        The true objective function
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

    # Initialize the model with available training data
    mll, model = build_surrogate_model(train_X, train_Y, bounds)
    for i in range(iterations if not SMOKE_TEST else 4):

        t0 = time.monotonic()

        # Fit model with new data: D =  D_old \cup D_new
        fit_gpytorch_mll(mll)

        # Use best_f (expected energy loss) observed so far
        ei = ExpectedImprovement(model, best_f=train_Y.min(), maximize=False)

        # Optimise and get new observation
        new_x, new_f = optimize_acqf_and_get_new_point(
            f, ic_generator, ei, bounds, nonlinear_inequality_constraints, SMOKE_TEST
        )  # TODO: add stoppping criterion

        # Update training points
        train_X = vstack([train_X, new_x])
        train_Y = vstack([train_Y, new_f])

        # Re-build model with new data, ready for fitting on next iteration
        mll, model = build_surrogate_model(train_X, train_Y, bounds)  # TODO: with state-dict here?

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


if __name__ == "__main__":

    vmec_input_files = []
    directory = "./src/vmec_input_files/nfp4/ours"
    for filename in os.listdir(directory):
        if filename.startswith("input.nfp4"):
            vmec_input_files.append(os.path.join(directory, filename))
    design = StellaratorDesign()
    train_X, train_Y, bounds, constraints = design.get_init_BO_params(vmec_input_files)

    SMOKE_TEST = True

    # Optimise
    run(design.f, design.sample_fake_Fourier_coefficients, train_X, train_Y, bounds, constraints, SMOKE_TEST)
