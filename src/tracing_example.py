import os
import sys

sys.path.insert(0, os.getcwd())
from mpi4py import MPI
import numpy as np
from torch import Tensor, tensor, cat, stack
from scipy.spatial import ConvexHull

from src.trace.trace_boozer import TraceBoozer


class StellaratorDesign:
    def __init__(self):

        # MPI stuff
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

        # Number of Fourier modes for optimization
        self.max_mode = 1
        self.d = 4 * self.max_mode**2 + 4 * self.max_mode

        # Target aspect ratio (ARIES-CS)
        self.aspect_target = 7.0

        # Fixed major radius (ARIES-CS size)
        self.major_radius = 1.7 * self.aspect_target

        # Volume average field strength
        self.target_volavgB = 1.0  # tesla

        # Tracing parameters
        self.s_label = 0.25  # surface label
        self.tmax = 1e-4  # max tracing time
        self.n_particles = 100  # number of particles

        # Tracing fidelity
        self.tracing_tol = 1e-8
        self.interpolant_degree = 3
        self.interpolant_level = 8
        self.bri_mpol = 8
        self.bri_ntor = 8

        # Output constraints on mirror ratio
        self.ns_B = 8  # maxB should be on boundary (so we could always just sample the boundary...)
        self.ntheta_B = 16
        self.nzeta_B = 16
        self.smin = 0.02
        self.smax = 1.0
        self.len_B_field_out = self.ns_B * self.ntheta_B * self.nzeta_B
        self.mirror_target = 1.35
        self.eps_B = (self.mirror_target - 1.0) / (self.mirror_target + 1.0)
        self.B_ub = self.target_volavgB * (1 + self.eps_B) * np.ones(self.len_B_field_out)  # upper bound, eq. 14
        self.B_lb = self.target_volavgB * (1 - self.eps_B) * np.ones(self.len_B_field_out)  # lower bound, eq. 14

    def build_tracer(self, vmec_input_file: str) -> TraceBoozer:
        return TraceBoozer(
            vmec_input_file,
            n_partitions=1,  # number of partitions for vmec (always use 1)
            max_mode=self.max_mode,  # maximum fourier modes for boundary
            major_radius=self.major_radius,  # major radius
            aspect_target=self.aspect_target,  # aspect ratio
            target_volavgB=self.target_volavgB,
            tracing_tol=self.tracing_tol,
            interpolant_degree=self.interpolant_degree,
            interpolant_level=self.interpolant_level,
            bri_mpol=self.bri_mpol,
            bri_ntor=self.bri_ntor,
            ns_B=self.ns_B,
            ntheta_B=self.ntheta_B,
            nzeta_B=self.nzeta_B,
            smin=self.smin,
            smax=self.smax,
        )

    @staticmethod
    def sample_fake_Fourier_coefficients(x: np.ndarray) -> np.ndarray:
        """
        Sample fake Fourier coefficients for testing purposes.

        Parameters
        ----------
        x0 : np.ndarray
            An initial (real) Fourier coefficient configuration

        Returns
        -------
        np.ndarray
            A corrupted version of x0 which simulates a new Fourier coefficient sample
        """
        noise = 0.1 * np.random.normal(0, 1, x.shape)
        return x + noise  # New Fourier coefficient 'sample'

    def f(self, x: np.ndarray) -> float:
        """
        Objective for minimization: expected energy loss f = E[3.5*np.exp(-2*c_times/tmax)]

        Parameters
        ----------
        x : np.ndarray
            vmec configuration variables [Fourier coefficients]

        Returns
        -------
        float
            Expected energy loss for the given configuration
        """
        # Sample particle positions (uniformly in theta, phi not in space)
        stz_inits, vpar_inits = self.tracer.sample_surface(self.n_particles, self.s_label)

        # Ensure compatibility with C++ tracing
        stz_inits = np.ascontiguousarray(stz_inits)
        vpar_inits = np.ascontiguousarray(vpar_inits)

        # Compute confinement times (heavy)
        c_times = self.tracer.compute_confinement_times(x, stz_inits, vpar_inits, self.tmax)

        if np.any(~np.isfinite(c_times)):
            # Vmec failed here; return worst possible value
            c_times = np.zeros(len(vpar_inits))

        # Energy retained by particle
        feat = 3.5 * np.exp(-2 * c_times / self.tmax)

        # Sample average
        res = np.mean(feat)
        loss_frac = np.mean(c_times < self.tmax)

        # Print with MPI
        if self.rank == 0:
            print("obj:", res, "P(loss):", loss_frac)
        sys.stdout.flush()

        return res

    def compute_B_field_vmec(self, x: np.ndarray, verbose: bool = False) -> np.ndarray:
        """
        Use VMEC to compute the |B| field.

        If VMEC fails, return an aray of zeros with length len_B_field_out

        x: Fourier variables array for VMEC.
        """
        # Compute modB on a grid
        modB = self.tracer.compute_modB_vmec(
            x, ns=self.ns_B, ntheta=self.ntheta_B, nphi=self.nzeta_B, smin=self.smin, smax=self.smax
        )

        # VMEC failure
        if len(modB) != self.len_B_field_out:
            return np.zeros(self.len_B_field_out)

        # print some stuff
        if self.rank == 0 and verbose:
            print("B-interval:", np.min(modB), np.max(modB))
            print("Mirror Ratio:", np.max(modB) / np.min(modB))

        return modB

    def get_B_field(self, x: np.ndarray, container: dict[float, float] = None) -> dict[np.ndarray, np.ndarray]:
        """
        Functions caches the B field values for a given x so that we don't have to recompute it.

        Parameters
        ----------
        x : np.ndarray
            Input
        container : dict[float, float]
            Container to store the B field values for a given x - if not given the B field will be computed and returned each time this function is called

        Returns
        -------
        dict[np.ndarray, np.ndarray]
            B field values for a given x
        """

        if container:
            if x not in container:
                container[x] = self.compute_B_field_vmec(x)
            return container[x]
        else:
            return self.compute_B_field_vmec(x)

    def acqf_nonlinear_inequality_constraints(self) -> list[tuple[callable, bool]]:
        """
        This function returns the nonlinear inequality constraints for the acquisition function. Nonlinear inequality constraints: equation (13) and (14) of [1], section 4.2.

        References
        ----------
        [1] Bindel, David, Matt Landreman, and Misha Padidar. "Direct optimization of fast-ion confinement in stellarators." Plasma Physics and Controlled Fusion 65.6 (2023): 065012.
        """

        # XXX: we could have separate constraints per dimension

        B_field_cache = {}  # Container to store already computed B field values
        B_diff_lower = lambda x: -(
            self.B_lb - self.get_B_field(x, B_field_cache)
        )  # Negated to conform to optimize_acqf docstring instructions
        B_diff_upper = lambda x: -(
            self.get_B_field(x, B_field_cache) - self.B_ub
        )  # Negated to conform to optimize_acqf docstring instructions

        return [(B_diff_lower, True), (B_diff_upper, True)]

    def calculate_bounds(self, train_X: Tensor) -> Tensor:
        """
        Function calculates the bounds for the input x based on the training data. Done by calculating the convex closure (convex hull) of all the input Fourier coefficients. This is a bit hand-wavy, but it's a start - the true region is almost surely not convex.

        Parameters
        ----------
        train_X : Tensor
            Input

        Returns
        -------
        Tensor
            Bounds for the Fourier coefficients
        """
        # TODO: implement this, we could also take x0, +/- 10% either way as the bounds

        # Compute the convex hull
        hull = ConvexHull(train_X)

        # Get the vertices of the convex hull
        hull_vertices = train_X[hull.vertices]

        # Compute min and max for each dimension
        min_values = np.min(hull_vertices, axis=0)
        max_values = np.max(hull_vertices, axis=0)

        assert len(min_values) == self.d
        assert len(max_values) == self.d

        return stack([tensor(min_values), tensor(max_values)])  # 2 x d

    def get_init_BO_params(
        self, input_files: str | list[str]
    ) -> tuple[Tensor, Tensor, Tensor, list[tuple[callable, bool]]]:
        """
        Function builds the initial training data for Bayesian optimization as well as the bounds and constraints, given a list of input files or a single input file.

        Parameters
        ----------
        input_files : str | list[str]
            Input files for the VMEC configuration

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, list[tuple[callable, bool]]]
            Training data, training labels, bounds, and constraints
        """

        # TODO: the GPR model requires us to scale the input features to the unit cube and standardize the output. We should do this here.

        if isinstance(input_files, str):
            # Build tracer for this input file
            self.tracer = self.build_tracer(input_files)
            # Sync seeds across MPI ranks
            self.tracer.sync_seeds()

            x0 = self.tracer.x0
            assert self.d == len(x0)  # Dimension of the input space (# of Fourier coefficients)
            train_X = tensor(x0).view(1, -1)  # 1 x d
            y0 = self.f(x0)
            train_Y = tensor([y0]).unsqueeze(-1)

        else:
            assert isinstance(input_files, list)
            train_X = None
            train_Y = None
            for file in input_files:
                # Build tracer for each input file
                self.tracer = self.build_tracer(file)

                # Features
                x0 = self.tracer.x0
                assert self.d == len(x0)  # Dimension of the input space (# of Fourier coefficients)
                if train_X is None:
                    train_X = tensor(x0).view(1, -1)  # 1 x d
                else:
                    train_X = cat((train_X, tensor(x0).view(1, -1)), dim=0)

                # Target
                y0 = self.f(x0)
                if train_Y is None:
                    train_Y = tensor([y0]).unsqueeze(-1)
                else:
                    train_Y = cat((train_Y, tensor([y0]).unsqueeze(-1)), dim=0)

        # TODO, fix InputDataWarning: Input data is not standardized (mean = tensor([2.3368], dtype=torch.float64), std = tensor([0.9414], dtype=torch.float64)). Please consider scaling the input to zero mean and unit variance.

        assert self.d == train_X.shape[1]  # Dimension of the input space (# of Fourier coefficients)

        bounds = self.calculate_bounds(train_X)
        constraints = self.acqf_nonlinear_inequality_constraints()
        return train_X, train_Y, bounds, constraints


if __name__ == "__main__":
    test = StellaratorDesign()
    vmec_input_file = "/Users/z004mktz/Code/fusion/alpha_particle_opt/src/vmec_input_files/input.nfp4_QH_cold_high_res"  # Used the cold start (input.nfp4_QH_cold_high_res) of this file instead of the default warm start (input.nfp4_QH_warm_start_high_res)
    vmec_input_file = [
        "./src/vmec_input_files/input.nfp4_QH_cold_high_res",
        "./src/vmec_input_files/input.nfp4_QH_cold_high_res_mirror_feasible",
        "./src/vmec_input_files/input.nfp4_QH_warm_start_high_res",
    ]
    vmec_input_file = "./src/vmec_input_files/input.test_misha"
    train_X, train_Y, bounds, constraints = test.get_init_BO_params(vmec_input_file)
