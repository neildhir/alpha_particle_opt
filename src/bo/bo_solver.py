from os import environ
import time
from functools import partial
import numpy as np
from typing import Optional, List
from numpy.typing import ArrayLike

from botorch.acquisition import ExpectedImprovement
from botorch.acquisition.analytic import LogExpectedImprovement
from torch import Tensor, vstack, save, from_numpy
from botorch.fit import fit_gpytorch_mll
from torch.autograd import grad as torch_grad
from scipy.optimize import OptimizeResult, minimize as scipy_minimize

from src.bo.initialisers import gen_batch_initial_conditions_nonlinear
from src.bo.bayes_opt import build_surrogate_model, optimize_acqf_and_get_new_point



class BoSolver:
    """
    A class implementing a BO solver that is compatible with optimization using 
    SIMOPT.
    """

    def __init__(self,
                 train_X: ArrayLike,
                 train_Y: ArrayLike,
                 max_iter: int = 50,
                 verbose: bool = True,
                 SMOKE_TEST: bool = False,
                 num_restarts: int = 8,
                 raw_samples: int = 32,
                 ):
        """
        Initialize the BO solver.

        Parameters
        ----------
        train_X : Array
            Input features (training data, Fourier coefficients)
        train_Y : Array
            Target (expected energy loss for this shape of the plasma boundary represented by the Fourier coefficients)
             shape (training data, 1)
        max_iter : int, optional
            Number of operations, by default 5 -- will be replaced with a stopping criterion
        verbose : bool, optional
            Print stuff or not, by default True
        SMOKE_TEST : bool, optional
            Whether we are in a testing mode or not.
        num_restarts : int, optional
            Number of multi-starts to use in optimizing the acquisition.
        raw_samples : int, optional
            The number of samples for initialization.
        """
        self.train_X = from_numpy(train_X)
        self.train_Y = from_numpy(train_Y)
        self.max_iter = max_iter
        self.verbose = verbose
        self.SMOKE_TEST = SMOKE_TEST
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples

    def sample_initial_points(self, acq_func: callable, bounds: Tensor,
                                  nonlinear_constraint: Optional[callable] = None):
        """
        Sample warm starts for optimizer.

        acq_func: callable,
            handle to the acquisition function.
        bounds: Tensor,
            2 x dim_x tensor of lower and then upper bounds.
        nonlinear_constraints: callable or None,
            A constraint function for constraints of the form c(x) >= 0. The function 
            has an argument X that takes in a 2d Tensor of points (n_points, dim_x)
            and returns an (n_points, 1) tensor of constraint evalautions. The constraint 
            must be single-valued -- if you have a vector valued constraint take the min.

        return
        --------
        X_cand: array, shape (num_restarts, dim_x)
            initial points for optimizer.
        """
        # TODO: expose q as a parameter that we force to be 1, so that shaping everywhere is consistent
        X_cand = gen_batch_initial_conditions_nonlinear(acq_func, q=1, num_restarts=self.num_restarts,            
                                                raw_samples=self.raw_samples,
                                                nonlinear_constraint=nonlinear_constraint,
                                                bounds=bounds
                                                ) # (num_restarts, q, dim_x)
        X_cand = X_cand.detach().numpy()
        X_cand = np.squeeze(X_cand, axis=1) # (num_restarts, dim_x)
        return X_cand

    def optimize_acquisition(self, objective: callable, 
                               X: Tensor, acq_func, bounds: List[tuple], constraints=[],
                                method="SLSQP", options: dict={},
                                ):
        """
        Generate samples and optimize the acquisition function

        Parameters
        ----------
        objective: callable, 
            function handle to the objective
        acq_func: 
            A torch acquisition function
        bounds: list 
            list of tuples of lower and upper bounds, i.e. [(0.0, 1.0), ..., (-1.0, 4.0)]
        constraints: list 
            containing any scipy.NonlinearConstraint and scipy.LinearConstraint instances.
        method: str,
            optimization method for the scipy minimize solver
        options: dict,
            dictionary of options to pass to the scipy minimize solver

        Return
        ----------
        new_x: Tensor,
            2d tensor of shape (num_restarts, dim_x) of new iterates.
        new_f: Tensor,
            2d tensor of shape (num_restarts, 1) of evaluations of new_x.
        """
        def acquisition_wrapper(x):
            # TODO: set up gradient with autodiff
            y = from_numpy(x).reshape((1,1,-1))
            return acq_func.forward(y).detach().numpy().item()

        # multi-start optimize the acquisition
        candidates = np.zeros(np.shape(X))
        for ii, x0 in enumerate(X):
            # TODO: set up joint optimization of all restarts
            # TODO: set up constraint gradients
            res = scipy_minimize(acquisition_wrapper, x0=x0, bounds=bounds,
                                 constraints=constraints,
                                 method=method, options=options)
            candidates[ii] = x0

        # Observe new values
        obj_candidates = np.array([objective(x) for x in candidates])

        return from_numpy(candidates), from_numpy(obj_candidates.reshape((-1,1)))


    def compound_nonlinear_constraints(self, X: Tensor, constraints=[]) -> Tensor:
        """
        Function to compute the compound nonlinear constraint for the acquisition function
        optimization.

        Parameters
        ----------
        X : Tensor
            2D array of candidate Fourier coefficients (candidates x # Fourier coefficients)

        Returns
        -------
        Tensor
            (candidates x 1) tensor of constraint values. One value per point that is 
            positive if the point is feasible, and negative otherwise.  
        """
        X = X.numpy()
        # reformat the nlc into c(x) >= 0
        acqf_nonlinear_inequality_constraints = [(lambda x: cc.fun(x) - cc.lb) for cc in constraints]
        acqf_nonlinear_inequality_constraints += [(lambda x: cc.ub - cc.fun(x)) for cc in constraints]

        nlc = np.zeros(len(X))
        for ii, xx in enumerate(X):
            # constraints are assumed to be of type c(x) >= 0
            cons = np.array([cc(xx) for cc in acqf_nonlinear_inequality_constraints]).flatten()
            nlc[ii] = np.min(cons)
        return from_numpy(nlc.reshape((-1, 1)))


    def solve(self, objective, x0, bounds, constraints, method="SLSQP", options={}):
        """
        Run the BO solver
        
        Parameters
        ----------
        objective: callable, 
            function handle to the objective
        x0: array, 
            incumbent solution
        bounds: list 
            list of tuples of lower and upper bounds, i.e. [(0.0, 1.0), ..., (-1.0, 4.0)]
        constraints: list 
            containing any scipy.NonlinearConstraint and scipy.LinearConstraint instances.
        method: str,
            optimization method for the scipy minimize solver
        options: dict,
            dictionary of options to pass to the scipy minimize solver

        Return 
        ----------
        result: OptimizeResult,
            An instance of scipy's OptimizeResult. Use
            result.x to get the optimal point and result.fun to to get the optimal objective value.
        """
        # TODO: we arent doing anything with x0 yet (include in evals?)
        
        train_X, train_Y = self.train_X, self.train_Y
        verbose = self.verbose

        # reformat the bounds for BO
        torch_bounds = from_numpy(np.array(bounds).T) # 2 x d

        # Build initial candidates generator for the acquisition function
        ic_nonlinear_inequality_constraints = partial(self.compound_nonlinear_constraints,
                                                      constraints=constraints)

        mll, model = build_surrogate_model(train_X, train_Y, torch_bounds)

        for i in range(self.max_iter):

            t0 = time.monotonic()

            # Fit model with new data: D =  D_old \cup D_new
            fit_gpytorch_mll(mll)

            # Use best_f (expected energy loss) observed so far
            ei = LogExpectedImprovement(model, best_f=train_Y.min(), maximize=False)

            X_cand = self.sample_initial_points(ei, bounds=torch_bounds,
                                                    nonlinear_constraint=ic_nonlinear_inequality_constraints)

            new_x, new_f = self.optimize_acquisition(objective, 
                                                       X=X_cand, 
                                                       acq_func=ei, 
                                                       bounds=bounds, 
                                                       constraints=constraints,
                                                       method=method, 
                                                       options=options,
                                                       )

            train_X = vstack([train_X, new_x])
            train_Y = vstack([train_Y, new_f])

            # Re-build model with new data, ready for fitting on next iteration
            mll, model = build_surrogate_model(
                train_X, train_Y, torch_bounds
            )  # TODO: we can prime the model with the state_dict insted of re-building it each time, faster.

            t1 = time.monotonic()
            best_f = train_Y.min().item()
            if self.verbose:
                print(
                    f"\nIteration {i}: best objective value = " f"{best_f:>4.2f}, " f"time = {t1-t0:>4.2f}.",
                )
            else:
                print(".", end="")

        # Save the results and found points with a unique name
        save_results = {
            "train_X": train_X,  # Fourier coefficients
            "train_Y": train_Y,  # Expected energy loss
            "model_state_dict": model.state_dict(),
            "mll_state_dict": mll.state_dict(),
        }
        timestamp = time.strftime("%Y%m%d%H%M%S")  # e.g. 20210909123456
        save(save_results, f"optimization_results_{timestamp}.pth")

        idx_opt = train_Y.argmin()
        result = OptimizeResult()
        result.x = train_X[idx_opt]
        result.fun = train_Y[idx_opt]
        result.train_X = train_X
        result.train_Y = train_Y
        return result


if __name__ == "__main__":
    from scipy.optimize import NonlinearConstraint
    
    dim = 2
    x0 = np.random.randn(dim)
    objective = lambda x: x @ x
    bounds = [(0, 10), (0,10)]
    constraints = [NonlinearConstraint(lambda x: x.sum(), 0.0, np.inf)]

    train_X = 10*np.random.uniform(size=(10,dim))
    train_Y = np.array([objective(x) for x in train_X]).reshape((-1,1))

    solver = BoSolver(train_X,
                 train_Y,
                 max_iter=10,
                 verbose=True
                 )   

    res = solver.solve(objective, x0, bounds, constraints, 
                       method='trust-constr', 
                       options={'maxiter':200})
    print(res)