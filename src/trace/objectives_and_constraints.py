import numpy as np

from simsopt._core import Optimizable
from simsopt.mhd import vmec_compute_geometry
from simsopt.field.boozermagneticfield import BoozerRadialInterpolant, InterpolatedBoozerField
from simsopt.field.tracing import (
    trace_particles_boozer,
    MaxToroidalFluxStoppingCriterion,
    MinToroidalFluxStoppingCriterion,
)

from src.utils.constants import ALPHA_PARTICLE_CHARGE, ALPHA_PARTICLE_MASS, FUSION_ALPHA_PARTICLE_ENERGY

def prepare_config(vmec, max_mode, major_radius, aspect_target, target_volavgB):
    """
    Prepare the configuration for optimization.
    This function rescales the configuration to a reactor scaling (major radius and toroidal flux). The 
    device is rescaled to achieve the specified major_radius. The toroidal flux is set to 
    approximate pi * a^2 * B, where 'a' is the minor radius and B is the volume average magnetic field.
    This function also sets the number of modes for the optimization and fixes the major radius.

    inputs
    vmec: instance of Vmec class.
    max_mode: int, maximum number of fourier modes.
    major_radius: float, fixed major radius.
    aspect_target: float, desired aspect ratio of the design.
    target_volavgB: float, target value for the volume average of |B|.

    return 
    vmec
    """
    surf = vmec.boundary

    # Set resolution
    mpol = ntor = max_mode if max_mode > 0 else surf.mpol
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=mpol, nmin=-ntor, nmax=ntor, fixed=False)
    
    # Rescale surface and set toroidal flux
    factor: float = major_radius / surf.get("rc(0,0)")
    surf.x *= factor
    surf.fix("rc(0,0)")

    target_avg_minor_rad: float = major_radius / aspect_target
    vmec.indata.phiedge = np.pi * (target_avg_minor_rad**2) * target_volavgB
    vmec.need_to_run_code = True

    vmec.surf = surf
    return vmec


class FastIonLoss(Optimizable):
    """
    An optimizable class for minimizing fast ion losses.
    """

    def __init__(
        self,
        vmec,
        mpi,
        sampler,
        tmax: float = 1e-4,
        tracing_tol: float = 1e-8,
        interpolant_degree: int = 3,
        interpolant_level: int = 8,
        bri_mpol: int = 32,
        bri_ntor: int = 32,
    ) -> None:
        """
        Initialize the TraceBoozer class.

        Parameters:
        ----------
        vmec: an instance of Vmec class.
        mpi: an instance of the MpiPartition class
        sampler: a function handle to sample particles. 
            ex.
            sampler = ParticleSampler.sample_surface or 
            sampler = FluxSurfaceGrid.surface_grid or 
        tmax: maximum tracing time in seconds, e.g. 1e-3.
        tracing_tol : float, default 1e-8
            Tolerance for determining tracing accuracy.
        interpolant_degree : int, default 3
            Degree of polynomial interpolants for field interpolation.
            1: fast but inaccurate, 3: slower but more accurate.
        interpolant_level : int, default 8
            Number of points per direction for Boozer radial interpolant.
            5: fast/inaccurate, 8: medium, 12: slow/accurate.
        bri_mpol, bri_ntor : int, default 32
            Number of poloidal and toroidal modes used in BoozXform.
            Lower values (e.g., 16) are faster.
        """

        self.mpi = mpi
        self.sampler = sampler
        self.tmax = tmax
        self.tracing_tol: float = tracing_tol
        self.interpolant_degree: int = interpolant_degree
        self.interpolant_level: int = interpolant_level
        self.bri_mpol: int = bri_mpol
        self.bri_ntor: int = bri_ntor

        # for caching
        self.need_to_run_code = True

        self.vmec = vmec
        super().__init__(depends_on=[vmec])

    def recompute_bell(self, parent=None):
        """
        This function will get called any time any of the DOFs of the
        parent class (vmec) change.
        """
        self.need_to_run_code = True
    
    def cache_results(self, confinement_times, is_success):
        """
        Cache the tracing results.
        """
        self.confinement_times = confinement_times
        self.is_success = is_success
        self.need_to_run_code = False

    def compute_boozer_field(self):
        """
        Use BoozXForm to compute the magnetic field in Boozer coordinates
        from a VMEC result.

        return: field, bri, is_success
        field: boozXform field. If VMEC fails, this will be None.
        bri: boozXform radial interpolant. If VMEC fails, this will be None.
        is_success: bool, True if VMEC succeeded.
        """
        is_success = True

        try:
            self.vmec.run()
        except:
            # VMEC failure!
            is_success = False
            return None, None, is_success

        # Construct radial interpolant of magnetic field
        bri = BoozerRadialInterpolant(
            equil=self.vmec, order=self.interpolant_degree, mpol=self.bri_mpol, ntor=self.bri_ntor, enforce_vacuum=True
        )

        # Construct 3D interpolation
        nfp = self.vmec.wout.nfp  # This is raising a false negative
        srange = (0, 1, self.interpolant_level)
        thetarange = (0, np.pi, self.interpolant_level)
        zetarange = (0, 2 * np.pi / nfp, self.interpolant_level)
        field = InterpolatedBoozerField(
            bri,
            degree=self.interpolant_degree,
            srange=srange,
            thetarange=thetarange,
            zetarange=zetarange,
            extrapolate=True,
            nfp=nfp,
            stellsym=True,
        )

        # cache results
        self.field = field
        self.bri = bri

        return field, bri, is_success


    def compute_confinement_times(self):
        """
        Compute the confinement times of particles in moving according to the vacuum GC
        approximation in Boozer coordinates.

        return:
            len(n_particles) array of confinement times.
            if the boozXform or vmec fail, then the array is filled
            with -np.inf.
        """
        is_success = True

        # get initial particle states
        stz_inits, vpar_inits = self.sampler()

        n_particles = len(vpar_inits)
        tmax = self.tmax
        fail_value = np.zeros(n_particles)

        # check cache
        if not self.need_to_run_code:
            return self.confinement_times, is_success

        field, bri, is_success = self.compute_boozer_field()

        if not is_success:
            # VMEC failure  
            confinement_times = fail_value
            self.cache_results(confinement_times, is_success)
            return fail_value, is_success

        stopping_criteria = [MaxToroidalFluxStoppingCriterion(0.99), MinToroidalFluxStoppingCriterion(0.01)]

        try:
            res_tys, res_zeta_hits = trace_particles_boozer(
                field,
                stz_inits,
                vpar_inits,
                tmax=tmax,
                mass=ALPHA_PARTICLE_MASS,
                charge=ALPHA_PARTICLE_CHARGE,
                Ekin=FUSION_ALPHA_PARTICLE_ENERGY,
                tol=self.tracing_tol,
                mode="gc_vac",
                comm=self.mpi.comm_groups,
                stopping_criteria=stopping_criteria,
                forget_exact_path=True,
            )
        except:
            # tracing failure
            is_success = False

        if not is_success:
            confinement_times = fail_value
            self.cache_results(confinement_times, is_success)
            return fail_value, is_success

        confinement_times = np.zeros(n_particles)
        for ii, res in enumerate(res_zeta_hits):

            # check if particle hit stopping criteria
            if len(res) > 0:
                if int(res[0, 1]) == -1:
                    # particle hit MaxToroidalFluxCriterion
                    confinement_times[ii] = res[0, 0]
                if int(res[0, 1]) == -2:
                    # particle hit MinToroidalFluxCriterion
                    confinement_times[ii] = tmax
            else:
                # didnt hit any stopping criteria
                confinement_times[ii] = tmax

        # cache the tracing results
        self.cache_results(confinement_times, is_success)

        return confinement_times, is_success
    
    def energy_loss(self):
        """
        Compute the energy lost due to electron collisions.
        This function can be used for optimization.
            
        If VMEC fails, 
            return the maximum value: 3.5
        Otherwise, 
            return E[3.5 * np.exp(-2 * c_times / tmax)]
        """
        c_times, is_success = self.compute_confinement_times()
        feat = 3.5 * np.exp(-2 * c_times / self.tmax)
        return np.mean(feat)
    
    def loss_fraction(self):
        """
        Compute the loss fraction
            P( c_times < tmax).
        This function can be used for optimization.
            
        If VMEC fails, 
            return the maximum value: 1.0
        Otherwise, 
            return the loss fraction
        """
        c_times, is_success = self.compute_confinement_times()
        return np.mean(c_times <  self.tmax)



class FieldStrength(Optimizable):
    """
    Optimiable class for computing the field strength |B|.
    """

    def __init__(self, vmec, ns=32, ntheta=32, nphi=32, smin=0.02, smax=1.0):
        """
        Parameters
        ----------
        x : np.ndarray
            vmec variable vector.
        ns : int, optional
            number of samples per dimension, by default 32
        ntheta : int, optional
            number of samples per dimension, by default 32
        nphi : int, optional
            number of samples per dimension, by default 32
        smin : float, optional
            values of the normalized toroidal flux, by default 0.02
        smax : float, optional
            values of the normalized toroidal flux, by default 1.0
        """
        self.ns = ns
        self.ntheta = ntheta
        self.nphi = nphi
        self.smin = smin 
        self.smax = smax
        self.output_size = ns*ntheta*nphi

        self.vmec = vmec
        Optimizable.__init__(self, depends_on=[vmec])

    def mirror_ratio(self):
        """
        Return the mirror ratio,
            max(B)/min(B).
        """
        B, is_success = self.compute()
        return np.max(B)/np.min(B), is_success
    
    def modB(self):
        """
        Compute the field strength. This function can be used for optimization.

        return
        modB: (ns*ntheta*nphi, ) array of modB values on the grid. If VMEC fails
            then the array is populated with zeros. 
        """
        B, is_success = self.compute()
        return B*is_success

    def compute(self) -> np.ndarray:
        """
        Compute |B| on a tensor product grid VMEC coordinates, (s, theta, phi).

        |B| is directly computed from the VMEC output in VMEC coordinates. 

        return: modB, is_success
        modB: (ns*ntheta*nphi, ) array of modB values on the grid. If is_success
            is False, then the array is populated with zeros. 
        is_success: bool, whether the computation is a success or not.
        """
        is_success = True

        # try to run vmec
        try:
            self.vmec.run()
        except:
            # VMEC failure!
            is_success = False
            return np.zeros(self.output_size), is_success

        nfp = self.vmec.wout.nfp
        s = np.linspace(self.smin, self.smax, self.ns)
        theta = np.linspace(0, 2 * np.pi, self.ntheta)
        phi = np.linspace(0, 2 * np.pi / nfp, self.nphi)

        # run vmec and compute the geometric quantites (VMEC may fail)
        data = vmec_compute_geometry(self.vmec, s, theta, phi)  # 3d array

        # return a 1d array
        modB = data.modB.flatten()
        return modB, is_success
