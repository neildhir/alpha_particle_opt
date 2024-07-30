import os
import sys

sys.path.insert(0, os.getcwd())
from mpi4py import MPI
import numpy as np
from torch import Tensor, tensor, cat, stack, load, from_numpy
from functools import cache
from scipy.spatial import ConvexHull

from src.trace.trace_boozer import TraceBoozer


class StellaratorDesign:
    def __init__(self, testing: bool = False):

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
        self.B_upper_limit = (
            self.target_volavgB * (1 + self.eps_B) * np.ones(self.len_B_field_out)
        )  # upper bound, eq. 14
        self.B_lower_limit = (
            self.target_volavgB * (1 - self.eps_B) * np.ones(self.len_B_field_out)
        )  # lower bound, eq. 14

        if testing:
            # Note absolute path
            vmec_input_file = "/Users/z004mktz/Code/fusion/alpha_particle_opt/src/vmec_input_files/nfp4/ours/input.nfp4_QH_cold_high_res"
            # Build tracer for this input file
            self.tracer = self.build_tracer(vmec_input_file)
            # Sync seeds across MPI ranks
            self.tracer.sync_seeds()

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

    # @cache
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

        # TODO: write snippet here which removes all VMEC-generated rubbish.

        return modB

    def B_lower_constraint(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the difference between B(x) and B_lb. To be valid should be smaller or equal to zero.

        Parameters
        ----------
        x : np.ndarray
            Input

        Returns
        -------
        np.ndarray
            Difference between B(x) and B_lb
        """
        return self.compute_B_field_vmec(x) - self.B_lower_limit  # >= 0

    def B_upper_constraint(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the difference between B_ub and B(x). To be valid should larger or equal to zero.

        Parameters
        ----------
        x : np.ndarray
            Input

        Returns
        -------
        np.ndarray
            Difference between B_ub and B(x)
        """
        return self.B_upper_limit - self.compute_B_field_vmec(x)  # >= 0

    def acqf_nonlinear_inequality_constraints(self) -> list[tuple[callable, bool]]:
        """
        This function returns the nonlinear (intra-point) inequality constraints for the acquisition function. Nonlinear inequality constraints: equation (13) and (14) of [1], section 4.2.

        References
        ----------
        [1] Bindel, David, Matt Landreman, and Misha Padidar. "Direct optimization of fast-ion confinement in stellarators." Plasma Physics and Controlled Fusion 65.6 (2023): 065012.
        """
        return [(self.B_lower_limit, True), (self.B_upper_limit, True)]

    def compound_nonlinear_constraint(self, X: Tensor) -> Tensor:
        """
        Function to compute the compound nonlinear constraint for the acquisition function.

        Parameters
        ----------
        X : Tensor
            2D array of candidate Fourier coefficients (candidates x # Fourier coefficients)

        Returns
        -------
        Tensor
            Valid points that satisfy the constraints
        """
        raise DeprecationWarning("This function is not used anymore.")

        # Compute B field for all points in X
        B_x = np.array([self.compute_B_field_vmec(x) for x in X.numpy()])
        # Create a mask for points that satisfy the constraints
        mask = np.logical_and(self.B_lower_limit <= B_x, B_x <= self.B_upper_limit)
        # Find points where all constraints are satisfied
        all_constraints_satisfied = np.all(mask, axis=1)
        # Filter valid points
        valid_points = X[all_constraints_satisfied]

        return from_numpy(valid_points)

    def compound_nonlinear_constraint_differentiable(self, X: Tensor) -> Tensor:
        """
        Function to compute the compound nonlinear constraint for the acquisition function.

        Parameters
        ----------
        X : Tensor
            2D array of candidate Fourier coefficients (candidates x # Fourier coefficients)

        Returns
        -------
        Tensor
            Valid points that satisfy the constraints
        """
        raise DeprecationWarning("This function is not used anymore.")

        # Compute B field for all points in X
        B_x = from_numpy(np.vstack([design.compute_B_field_vmec(x) for x in train_X]))

        B_lower_diff = B_x - self.B_lower_limit  # >= 0
        B_upper_diff = self.B_upper_limit - B_x  # >= 0

        return stack((B_lower_diff, B_upper_diff), dim=-1).min(-1).values

    def calculate_hull_bounds(self, train_X: Tensor) -> Tensor:
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

    def calculate_bounds(self, train_X: Tensor) -> Tensor:
        """
        Function calculates the box bounds for the input x based on the training data.

        Parameters
        ----------
        train_X : Tensor
            Input

        Returns
        -------
        Tensor
            Bounds for the Fourier coefficients
        """

        # Compute min and max for each dimension
        min_values = train_X.min(dim=0)
        max_values = train_X.max(dim=0)

        # Add 10% slack to the bounds
        return stack([0.9 * min_values.values, 1.1 * max_values.values])  # 2 x d

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

        try:
            bo_params = load("bo_params.pth")
            train_X = bo_params["train_X"]
            train_Y = bo_params["train_Y"]

        except FileNotFoundError:

            assert len(input_files) > 0, "No input files provided."

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
                    print("\nProcessing file:", file)
                    # Build tracer for each input file
                    self.tracer = self.build_tracer(file)
                    # Sync seeds across MPI ranks
                    self.tracer.sync_seeds()

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
                    print("Finished.")
                    continue

            assert self.d == train_X.shape[1]  # Dimension of the input space (# of Fourier coefficients)

        bounds = self.calculate_bounds(train_X)
        constraints = self.acqf_nonlinear_inequality_constraints()
        # constraints = self.compound_nonlinear_constraint

        return train_X, train_Y, bounds, constraints


if __name__ == "__main__":
    vmec_input_files = []
    directory = "./src/vmec_input_files/nfp4/ours"
    for filename in os.listdir(directory):
        if filename.startswith("input.nfp4"):
            vmec_input_files.append(os.path.join(directory, filename))
    design = StellaratorDesign()
    train_X, train_Y, bounds, constraints = design.get_init_BO_params(vmec_input_files)
