from os import environ
import time
from functools import partial
import numpy as np
from typing import Optional, List
from numpy.typing import ArrayLike

from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.acquisition import ExpectedImprovement
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.analytic import LogExpectedImprovement, LogConstrainedExpectedImprovement
from torch import Tensor, vstack, save, from_numpy, rand as torch_rand, topk, float64
from botorch.fit import fit_gpytorch_mll
from torch.autograd import grad as torch_grad
from botorch.optim import optimize_acqf

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
        train_Y : 2d, Array
            First column is the objective evaluations: (expected energy loss for this shape of the plasma boundary represented by the Fourier coefficients)
            Remaining columns are constraint evaluations: (field strength evaluation). There can be any number
            of constraints, however, we will be building GP models of each one. So, it is best for computational complexity 
            to have less constraints. The constraints should be passed in the same order to the solve() function.
            shape (training data, n_constraints)
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
        self.q = 1 # q=1 for EI acqf

        self.n_obj_con = np.shape(train_Y)[1] 
        self.n_constraints = self.n_obj_con - 1
        # index of objective/constraints in train_Y
        self.objective_index = 0
        self.constraint_indexes = range(1, self.n_obj_con)


    def build_singleTaskGP(self, train_X, train_Y, bounds):
        """
        Build a GP model for the objectives and constraints. 
        It is assumed that the objective and constraints are evaluted
        at the same set of points.

        inputs
        ------
        train_X: torch tensor, shape (n, dim_x)
        train_Y: torch tensor, shape (n, n_constraints)
        bounds: torch tensor, shape (2, dim_x)
            bound constraints on x.
        
        return
        ------
        mll: marginal log likelihood.
        model: GP model.
        """

        # TODO: use a noise model for the modB constraints
        # Build surrogate model, kernel uses ARD by default
        model = SingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            outcome_transform=Standardize(m=self.n_obj_con),
            input_transform=Normalize(d=train_X.size(1), bounds=bounds),
        )

        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        return mll, model

    # def sample_initial_points(self, acq_func: callable, bounds: Tensor):
    #     """
    #     Sample warm starts for optimizer.

    #     acq_func: callable,
    #         handle to the acquisition function.
    #     bounds: Tensor,
    #         2 x dim_x tensor of lower and then upper bounds.

    #     return
    #     --------
    #     X_cand: array, shape (num_restarts, dim_x)
    #         initial points for optimizer.
    #     """
    #     # TODO: use rejection sampling here or the precomputed samples

    #     # uniformly sample the region
    #     dim_x = bounds.shape[1]
    #     unif = torch_rand((self.raw_samples, 1, dim_x)) # (raw_samples, q, dim_x)
    #     X_cand =  (bounds[1] - bounds[0])*unif + bounds[0] # (raw_samples, q, dim_x)

    #     # TODO: test this
    #     # evaluate the acq_func
    #     qx = acq_func(X_cand) # (raw_samples, 1)
    #     _, top_idx = topk(-qx, self.num_restarts)  
    #     X_cand = X_cand[top_idx] # (num_restarts, q, dim_x)

    #     X_cand = X_cand.detach().numpy()
    #     X_cand = np.squeeze(X_cand, axis=1) # (num_restarts, dim_x)
    #     return X_cand

    # def optimize_acquisition(self, acqf, bounds):
    #     """
    #     Generate samples and optimize the acquisition function

    #     Parameters
    #     ----------
    #     acqf: acquisition function
    #     bounds: bound constraints for torch.

    #     Return
    #     ----------
    #     new_x: Tensor,
    #         2d tensor of shape (num_restarts, dim_x) of new iterates.
    #     new_f: Tensor,
    #         2d tensor of shape (num_restarts, 1) of evaluations of new_x.
    #     """
    #     # (q, dim_x)
    #     candidates, _ = optimize_acqf(acqf, bounds, q=1,
    #                                     num_restarts=self.num_restarts,
    #                                     raw_samples=self.raw_samples) 

    #     return candidates 

    def evaluate_candidates(self, X, objective, constraints):
        """
        Evaluate the objective and constraint values of the candidates, X. 

        inputs
        ------
        X: torch tensor, shape (n_points, dim_x)
        
        return
        ------
        Y: torch tensor, shape (n_points, n_obj_con)
        """
        n_x = X.shape[0]
        Y = np.zeros((n_x, self.n_obj_con))

        for ii, x in enumerate(X):
            x = x.numpy().flatten()

            Y[ii, self.objective_index] = objective(x)

            if self.n_constraints > 0:
                Y[ii, self.constraint_indexes] = [cc.fun(x) for cc in constraints]
                
        Y = from_numpy(Y.reshape((-1, self.n_obj_con)))

        return Y

    # def compound_nonlinear_constraints(self, X: Tensor, constraints=[]) -> Tensor:
    #     """
    #     Function to compute the compound nonlinear constraint for the acquisition function
    #     optimization.

    #     Parameters
    #     ----------
    #     X : Tensor
    #         2D array of candidate Fourier coefficients (candidates x # Fourier coefficients)

    #     Returns
    #     -------
    #     Tensor
    #         (candidates x 1) tensor of constraint values. One value per point that is 
    #         positive if the point is feasible, and negative otherwise.  
    #     """
    #     X = X.numpy()
    #     # reformat the nlc into c(x) >= 0
    #     acqf_nonlinear_inequality_constraints = [(lambda x: cc.fun(x) - cc.lb) for cc in constraints]
    #     acqf_nonlinear_inequality_constraints += [(lambda x: cc.ub - cc.fun(x)) for cc in constraints]

    #     nlc = np.zeros(len(X))
    #     for ii, xx in enumerate(X):
    #         # constraints are assumed to be of type c(x) >= 0
    #         cons = np.array([cc(xx) for cc in acqf_nonlinear_inequality_constraints]).flatten()
    #         nlc[ii] = np.min(cons)
    #     return from_numpy(nlc.reshape((-1, 1)))


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

        # reformat the bounds for BO
        torch_bounds = from_numpy(np.array(bounds).T) # 2 x d

        # constraints for optimizing the acquisition function
        acqf_constraints = {}
        for ii, cc in enumerate(constraints):
            idx = self.constraint_indexes[ii]
            lb = cc.lb if np.isfinite(cc.lb) else None
            ub = cc.ub if np.isfinite(cc.ub) else None
            acqf_constraints[idx] = [lb, ub]

        mll, model = self.build_singleTaskGP(train_X, train_Y, torch_bounds)

        # TODO: remove
        # model = SingleTaskGP(
        #     train_X=train_X,
        #     train_Y=train_Y[:,:1],
        #     outcome_transform=Standardize(m=1),
        #     input_transform=Normalize(d=train_X.size(1), bounds=torch_bounds),
        # )
        # mll = ExactMarginalLogLikelihood(model.likelihood, model)

        for i in range(self.max_iter):
            t0_loop = time.monotonic()

            if self.verbose:
                print('\nFitting model:')
            t0 = time.monotonic()

            # Fit model with new data
            fit_gpytorch_mll(mll)

            t1 = time.monotonic()
            if self.verbose:
                print('--> time', t1 - t0)

            # Use best_f (expected energy loss) observed so far
            acqf = LogConstrainedExpectedImprovement(model, best_f=train_Y[:,self.objective_index].min(),
                                                     objective_index = self.objective_index,
                                                     constraints = acqf_constraints,
                                                     maximize=False)
            # acqf = ExpectedImprovement(model, best_f=train_Y[:,self.objective_index].min(),
            #                                          maximize=False)

            if self.verbose:
                print('Optimizing acq_func:')
            t0 = time.monotonic()

            new_x, _ = optimize_acqf(acqf, torch_bounds.float(), q=self.q,
                                            num_restarts=self.num_restarts,
                                            raw_samples=self.raw_samples) # (q, dim_x)
            
            t1 = time.monotonic()
            if self.verbose:
                print('--> time', t1 - t0)

            if self.verbose:
                print('Evaluating candidate points:')
            t0 = time.monotonic()

            new_f = self.evaluate_candidates(new_x, objective, constraints)

            t1 = time.monotonic()
            if self.verbose:
                print('--> time', t1 - t0)

            # update the dataset
            train_X = vstack([train_X, new_x])
            train_Y = vstack([train_Y, new_f])

            # Re-build model with new data, ready for fitting on next iteration
            mll, model = self.build_singleTaskGP(train_X, train_Y, torch_bounds) # TODO: we can prime the model with the state_dict insted of re-building it each time, faster.

            t1_loop = time.monotonic()
            best_f = train_Y[:,self.objective_index].min().item()
            if self.verbose:
                print(
                    f"\nIteration {i}: best objective value = " f"{best_f:>4.4f}, " f"time = {t1_loop-t0_loop:>4.2f}.",
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

        idx_opt = train_Y[:,self.objective_index].argmin()
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
    objective = lambda x: (x - 1.2) @ (x-1.2) #+ 0.3*np.sin(x @ x)
    constraint = lambda x: x.sum()
    bounds = [(0, 10), (0,10)]

    constraints = [NonlinearConstraint(constraint, -np.inf, 1.0)]

    train_X = 10*np.random.uniform(size=(10,dim))
    train_Y = np.array([[objective(x), constraint(x)] for x in train_X]).reshape((-1,2))

    print(train_X)
    print(train_Y)

    solver = BoSolver(train_X,
                 train_Y,
                 max_iter=20,
                 verbose=True
                 )

    res = solver.solve(objective, x0, bounds, constraints, 
                       method='SLSQP', options={'maxiter':20, 'ftol':1e0})
    print("")
    print(res)
