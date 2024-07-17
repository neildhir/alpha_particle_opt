from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from torch import Tensor


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


def optimize_acqf_and_get_new_point(
    f: callable,
    acq_func: ExpectedImprovement,
    bounds: Tensor,
    nonlinear_inequality_constraints: list[tuple[callable, bool]],
    SMOKE_TEST: bool,
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
    nonlinear_inequality_constraints : list[tuple[callable, bool]]
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

    NUM_RESTARTS = 10 if not SMOKE_TEST else 2
    RAW_SAMPLES = 512 if not SMOKE_TEST else 32

    stopping_criterion = None  # TODO: implement
    # acqf = qExpectedImprovement(model=m1, best_f=0.0)
    # opt_inputs = OptimizeAcqfInputs(
    # acq_function=acqf, bounds=bounds, q=1, num_restarts=1, **kwargs
    # )
    # ic_generator = opt_inputs.get_ic_generator()

    # Initial condition (IC) generation is a fairly unsupported feature in BoTorch (at the time or writing). For details see: https://github.com/pytorch/botorch/issues/1572

    candidates, _ = optimize_acqf(
        ic_generator=None,  # TODO: have to provide this
        acq_function=acq_func,
        bounds=bounds,
        nonlinear_inequality_constraints=nonlinear_inequality_constraints,
        num_restarts=NUM_RESTARTS,  # XXX: perhaps reduce a spot
        raw_samples=RAW_SAMPLES,  # XXX: perhaps reduce a spot
        q=1,  # Explore methods which allow q > 1
    )
    # observe new values
    new_x = candidates.detach()  # Detach to avoid gradient updates
    new_obj = f(new_x.numpy().flatten())  # This is cumbersome, re-write

    return new_x, new_obj


def initial_candidate_generator(n: int, **kwargs) -> Tensor:
    # Generate n initial conditions that satisfy your constraints
    # This is a simple example; you'll need to adapt it to your specific constraints
    samples = torch.rand(n, d)  # d is the dimension of your search space
    while not all(constraint(samples) <= 0):
        samples = torch.rand(n, d)
    return samples
