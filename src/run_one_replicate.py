from os import environ

from torch import Tensor, vstack, save
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement
import time
from functools import partial

from src.examples.tracing_example import StellaratorDesign
from src.bo.initialisers import gen_batch_initial_conditions_nonlinear
from src.bo.bayes_opt import build_surrogate_model, optimize_acqf_and_get_new_point
import os

SMOKE_TEST = environ.get("SMOKE_TEST")  # TODO: finish this


def run(
    f: callable,
    train_X: Tensor,
    train_Y: Tensor,
    bounds: Tensor,
    ic_nonlinear_inequality_constraints: callable,
    acqf_nonlinear_inequality_constraints: callable,
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
    nonlinear_inequality_constraints : callable
        Magnetic field strength constraints as a function of x
    iterations : int, optional
        Number of operations, by default 5 -- will be replaced with a stopping criterion
    verbose : bool, optional
        Print stuff or not, by default True
    """

    # Initialize the model with available training data
    mll, model = build_surrogate_model(train_X, train_Y, bounds)

    # Build initial candidates generator for the acquisition function
    _gen_batch_initial_conditions_nonlinear = partial(
        gen_batch_initial_conditions_nonlinear,
        nonlinear_constraint=ic_nonlinear_inequality_constraints,
    )

    for i in range(4 if SMOKE_TEST else iterations):

        t0 = time.monotonic()

        # Fit model with new data: D =  D_old \cup D_new
        fit_gpytorch_mll(mll)

        # Use best_f (expected energy loss) observed so far
        ei = ExpectedImprovement(model, best_f=train_Y.min(), maximize=False)

        # Optimise and get new observation
        new_x, new_f = optimize_acqf_and_get_new_point(
            f=f,
            ic_generator=_gen_batch_initial_conditions_nonlinear,
            acq_func=ei,
            bounds=bounds,
            constraint=acqf_nonlinear_inequality_constraints,
            SMOKE_TEST=SMOKE_TEST,
        )

        # Update training points
        train_X = vstack([train_X, new_x])
        train_Y = vstack([train_Y, new_f])

        # Re-build model with new data, ready for fitting on next iteration
        mll, model = build_surrogate_model(
            train_X, train_Y, bounds
        )  # TODO: we can prime the model with the state_dict insted of re-building it each time, faster.

        t1 = time.monotonic()
        best_f = train_Y.min().item()
        if verbose:
            print(
                f"\nIteration {i}: best objective value = " f"{best_f:>4.2f}, " f"time = {t1-t0:>4.2f}.",
                end="",
            )
        else:
            print(".", end="")

    # Save the results and found points with a unique name
    results = {
        "train_X": train_X,  # Fourier coefficients
        "train_Y": train_Y,  # Expected energy loss
        "model_state_dict": model.state_dict(),
        "mll_state_dict": mll.state_dict(),
    }
    timestamp = time.strftime("%Y%m%d%H%M%S")  # e.g. 20210909123456
    save(results, f"optimization_results_{timestamp}.pth")


if __name__ == "__main__":

    vmec_input_files = []
    directory = "./src/vmec_input_files/nfp4/ours"
    for filename in os.listdir(directory):
        if filename.startswith("input.nfp4"):
            vmec_input_files.append(os.path.join(directory, filename))
    design = StellaratorDesign(testing=True)
    train_X, train_Y, bounds = design.get_init_BO_params(vmec_input_files)

    SMOKE_TEST = True

    # Optimise
    run(
        f=design.f,
        train_X=train_X,
        train_Y=train_Y,
        bounds=bounds,
        ic_nonlinear_inequality_constraints=design.compound_nonlinear_constraint_differentiable,  # Compound intra-point constraint
        acqf_nonlinear_inequality_constraints=design.acqf_nonlinear_inequality_constraints(),  # Intra-point constraints
        SMOKE_TEST=SMOKE_TEST,
    )
