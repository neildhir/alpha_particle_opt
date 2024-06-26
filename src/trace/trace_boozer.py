import numpy as np
from simsopt.field.boozermagneticfield import BoozerRadialInterpolant, InterpolatedBoozerField
from simsopt.field.tracing import (
    trace_particles_boozer,
    MaxToroidalFluxStoppingCriterion,
    MinToroidalFluxStoppingCriterion,
)
from simsopt.util.mpi import MpiPartition
from simsopt.mhd import Vmec, vmec_compute_geometry
from mpi4py import MPI

from src.sample.radial_density import RadialDensity
from src.utils.constants import *  # TODO: let's redo this, this is bad practice (import all)
from src.utils.divide_work import *  # TODO: let's redo this, this is bad practice (import all)


class TraceBoozer:
    """
    A class to make tracing from a vmec configuration a bit
    more convenient.
    """

    def __init__(
        self,
        vmec_input,
        n_partitions=1,
        max_mode=-1,
        major_radius=13.6,
        aspect_target=8.0,
        target_volavgB=5.0,
        tracing_tol=1e-8,
        interpolant_degree=3,
        interpolant_level=8,
        bri_mpol=32,
        bri_ntor=32,
    ):
        """
        vmec_input: vmec input file
        n_partitions: number of partitions used by vmec mpi.
        max_mode: number of modes used by vmec
        major_radius: will rescale entire device so that this is the major radius.
                  If the surface is purely a torus, then setting the major and minor radius
                  like this will give you the right aspect ratio. But if the surface
                  is not a normal torus, then the aspect ratio may be much smaller or larger.
        target_volavgB: will set phiedge so that this is the volume averge |B|.
                        phiedge= pi* a^2 * B approximately, so we try to rescale to
                        achieve the target B value.
        tracing_tol:a tolerance used to determine the accuracy of the tracing
        interpolant_degree: degree of the polynomial interpolants used for
          interpolating the field. 1 is fast but innacurate, 3 is slower but more accurate.
        interpolant_level: number of points used to interpolate the boozer radial
          interpolant (per direction). 5=fast/innacurate, 8=medium, 12=slow/accurate
        bri_mpol,bri_ntor: number of poloidal and toroidal modes used in BoozXform,
            less modes. 16 is faster than 32.
        """

        self.vmec_input = vmec_input
        self.max_mode = max_mode

        # For RZFourier rep
        self.mpi = MpiPartition(n_partitions)
        self.vmec = Vmec(vmec_input, mpi=self.mpi, keep_all_files=False, verbose=False)
        # get the boundary rep
        self.surf = self.vmec.boundary

        if max_mode < 0:
            mpol = self.surf.mpol
            ntor = self.surf.ntor
        else:
            mpol = max_mode
            ntor = max_mode

        # if len(x0) > 0:
        #  """
        #  Load a point with. We assume that x0_max_mode <= max_mode, that
        #  the major radius of the configuration was fixed and is not represented
        #  in the array x0.
        #  We assume that the toroidal flux was rescaled by target_volavgB.
        #  """
        #  assert x0_max_mode <= max_mode,"we cannot decrease the max_mode"
        #  # set up the boundary representation for x0
        #  self.surf.fix_all()
        #  self.surf.fixed_range(mmin=0, mmax=x0_max_mode,
        #                   nmin=-x0_max_mode, nmax=x0_max_mode, fixed=False)

        #  # rescale the vmec_input point to the major radius
        #  factor = major_radius/self.surf.get("rc(0,0)")
        #  self.surf.x = self.surf.x*factor
        #  self.surf.set("rc(0,0)",major_radius)
        #  self.surf.fix("rc(0,0)") # fix the Major radius

        #  # set the toroidal flux based off the vmec input, not the current point
        #  #avg_minor_rad = self.surf.get('rc(0,0)')/self.surf.aspect_ratio() # true avg minor radius
        #  target_avg_minor_rad = major_radius/aspect_target # target avg minor radius
        #  self.vmec.indata.phiedge = np.pi*(target_avg_minor_rad**2)*target_volavgB
        #  self.vmec.need_to_run_code = True

        #  # now set x0 as the boundary
        #  self.surf.x = np.copy(x0)

        # set the desired resolution
        self.surf.fix_all()
        self.surf.fixed_range(mmin=0, mmax=mpol, nmin=-ntor, nmax=ntor, fixed=False)

        # rescale the surface by the major radius; if we havent already.
        factor = major_radius / self.surf.get("rc(0,0)")
        self.surf.x = self.surf.x * factor

        # fix the major radius
        self.surf.fix("rc(0,0)")

        # rescale the toroidal flux; if we havent already
        # avg_minor_rad = self.surf.get('rc(0,0)')/self.surf.aspect_ratio() # true avg minor radius
        target_avg_minor_rad = major_radius / aspect_target  # target avg minor radius
        self.vmec.indata.phiedge = np.pi * (target_avg_minor_rad**2) * target_volavgB
        self.vmec.need_to_run_code = True
        # self.vmec.run()
        # print('aspect',self.vmec.aspect())
        # print('volavgB',self.vmec.wout.volavgB)
        # print('phiedge',self.vmec.indata.phiedge)

        # variables
        self.x0 = np.copy(self.surf.x)  # nominal starting point
        self.dim_x = len(self.x0)  # dimension

        # tracing params
        self.tracing_tol = tracing_tol
        self.interpolant_degree = interpolant_degree
        self.interpolant_level = interpolant_level
        self.bri_mpol = bri_mpol
        self.bri_ntor = bri_ntor

        # placeholders
        self.x_field = np.zeros(self.dim_x)
        self.field = None
        self.bri = None

    def expand_x(self, max_mode):
        """
        Expands the parameter space to the desired max mode.

        return the current point in the higher dim space.
        """
        # Define parameter space:
        self.surf.fix_all()
        self.surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
        self.surf.fix("rc(0,0)")  # Major radius
        self.x0 = np.copy(self.surf.x)  # nominal starting point
        self.dim_x = len(self.x0)  # dimension
        self.x_field = np.zeros(self.dim_x)
        return np.copy(self.x0)

    def sync_seeds(self, sd=None):
        """
        Sync the np.random.seed of the various worker groups.
        The seed is a random number <1e6.
        """
        # only sync across mpi group
        seed = np.zeros(1)
        # if self.mpi.proc0_world:
        if self.mpi.proc0_groups:
            if sd is not None:
                seed = sd * np.ones(1)
            else:
                seed = np.random.randint(int(1e6)) * np.ones(1)
        # self.mpi.comm_world.Bcast(seed,root=0)
        self.mpi.comm_groups.Bcast(seed, root=0)
        np.random.seed(int(seed[0]))
        return int(seed[0])

    def radial_grid(self, ns, ntheta, nzeta, nvpar, min_cdf=0.01, max_cdf=0.95, vpar_lb=-V_MAX, vpar_ub=V_MAX):
        """
        Build a 4d grid over the flux coordinates and vpar which is uniform in the
        radial CDF.
        min_cdf,max_cdf: lower and upper bounds on the grid in CDF space
        """
        # uniformly grid according to the radial measure
        sampler = RadialDensity(1000)
        surfaces = np.linspace(min_cdf, max_cdf, ns)
        surfaces = sampler._cdf_inv(surfaces)
        # use fixed particle locations
        thetas = np.linspace(0, 2 * np.pi, ntheta)
        zetas = np.linspace(0, 2 * np.pi / self.surf.nfp, nzeta)
        # vpars = symlog_grid(vpar_lb,vpar_ub,nvpar)
        vpars = np.linspace(vpar_lb, vpar_ub, nvpar)
        # build a mesh
        [surfaces, thetas, zetas, vpars] = np.meshgrid(surfaces, thetas, zetas, vpars)
        stz_inits = np.zeros((ns * ntheta * nzeta * nvpar, 3))
        stz_inits[:, 0] = surfaces.flatten()
        stz_inits[:, 1] = thetas.flatten()
        stz_inits[:, 2] = zetas.flatten()
        vpar_inits = vpars.flatten()
        return stz_inits, vpar_inits

    def flux_grid(self, ns, ntheta, nzeta, nvpar, s_min=0.01, s_max=1.0, vpar_lb=-V_MAX, vpar_ub=V_MAX):
        """
        Build a 4d grid over the flux coordinates and vpar.
        """
        # use fixed particle locations
        surfaces = np.linspace(s_min, s_max, ns)
        thetas = np.linspace(0, 2 * np.pi, ntheta)
        zetas = np.linspace(0, 2 * np.pi / self.surf.nfp, nzeta)
        # vpars = symlog_grid(vpar_lb,vpar_ub,nvpar)
        vpars = np.linspace(vpar_lb, vpar_ub, nvpar)
        # build a mesh
        [surfaces, thetas, zetas, vpars] = np.meshgrid(surfaces, thetas, zetas, vpars)
        stz_inits = np.zeros((ns * ntheta * nzeta * nvpar, 3))
        stz_inits[:, 0] = surfaces.flatten()
        stz_inits[:, 1] = thetas.flatten()
        stz_inits[:, 2] = zetas.flatten()
        vpar_inits = vpars.flatten()
        return stz_inits, vpar_inits

    def surface_grid(self, s_label, ntheta, nzeta, nvpar, vpar_lb=-V_MAX, vpar_ub=V_MAX):
        """
        Builds a grid on a single surface.
        """
        # use fixed particle locations
        # theta is [0,pi] with stellsym
        thetas = np.linspace(0, 2 * np.pi, ntheta)
        zetas = np.linspace(0, 2 * np.pi / self.surf.nfp, nzeta)
        # vpars = symlog_grid(vpar_lb,vpar_ub,nvpar)
        vpars = np.linspace(vpar_lb, vpar_ub, nvpar)
        # build a mesh
        [thetas, zetas, vpars] = np.meshgrid(thetas, zetas, vpars)
        stz_inits = np.zeros((ntheta * nzeta * nvpar, 3))
        stz_inits[:, 0] = s_label
        stz_inits[:, 1] = thetas.flatten()
        stz_inits[:, 2] = zetas.flatten()
        vpar_inits = vpars.flatten()
        return stz_inits, vpar_inits

    def poloidal_grid(self, zeta_label, ns, ntheta, nvpar, s_max=0.98):
        """
        Builds a grid on a poloidal cross section
        """
        # bounds
        vpar_lb = np.sqrt(FUSION_ALPHA_SPEED_SQUARED) * (-1)
        vpar_ub = np.sqrt(FUSION_ALPHA_SPEED_SQUARED) * (1)
        # use fixed particle locations
        surfaces = np.linspace(0.01, s_max, ns)
        thetas = np.linspace(0, 2 * np.pi, ntheta)
        vpars = np.linspace(vpar_lb, vpar_ub, nvpar)
        # build a mesh
        [surfaces, thetas, vpars] = np.meshgrid(surfaces, thetas, vpars)
        stz_inits = np.zeros((ns * ntheta * nvpar, 3))
        stz_inits[:, 0] = surfaces.flatten()
        stz_inits[:, 1] = thetas.flatten()
        stz_inits[:, 2] = zeta_label
        vpar_inits = vpars.flatten()
        return stz_inits, vpar_inits

    def sample_volume(self, n_particles):
        """
        Sample the volume using the radial density sampler
        """

        # SAA sampling over (s,theta,zeta,vpar)
        s_inits = np.zeros(n_particles)
        theta_inits = np.zeros(n_particles)
        zeta_inits = np.zeros(n_particles)
        vpar_inits = np.zeros(n_particles)
        # if rank == 0:
        if self.mpi.proc0_groups:
            sampler = RadialDensity(1000)
            s_inits = sampler.sample(n_particles)
            # randomly sample theta,zeta,vpar
            # theta is [0,pi] with stellsym
            theta_inits = np.random.uniform(0, 2 * np.pi, n_particles)
            zeta_inits = np.random.uniform(0, 2 * np.pi / self.surf.nfp, n_particles)
            vpar_inits = np.random.uniform(-V_MAX, V_MAX, n_particles)
        # broadcast the points
        self.mpi.comm_groups.Bcast(s_inits, root=0)
        self.mpi.comm_groups.Bcast(theta_inits, root=0)
        self.mpi.comm_groups.Bcast(zeta_inits, root=0)
        self.mpi.comm_groups.Bcast(vpar_inits, root=0)
        # stack the samples
        stp_inits = np.vstack((s_inits, theta_inits, zeta_inits)).T
        return stp_inits, vpar_inits

    def sample_surface(self, n_particles, s_label):
        """
        Sample the volume using the radial density sampler
        """
        ## divide the particles
        # comm = MPI.COMM_WORLD
        # size = comm.Get_size()
        # rank = comm.Get_rank()
        # SAA sampling over (theta,zeta,vpar) for a fixed surface
        s_inits = s_label * np.ones(n_particles)
        theta_inits = np.zeros(n_particles)
        zeta_inits = np.zeros(n_particles)
        vpar_inits = np.zeros(n_particles)
        # if rank == 0:
        if self.mpi.proc0_groups:
            # randomly sample theta,zeta,vpar
            # theta is [0,pi] with stellsym
            theta_inits = np.random.uniform(0, 2 * np.pi, n_particles)
            zeta_inits = np.random.uniform(0, 2 * np.pi / self.surf.nfp, n_particles)
            vpar_inits = np.random.uniform(-V_MAX, V_MAX, n_particles)
        # broadcast the points
        self.mpi.comm_groups.Bcast(theta_inits, root=0)
        self.mpi.comm_groups.Bcast(zeta_inits, root=0)
        self.mpi.comm_groups.Bcast(vpar_inits, root=0)
        # stack the samples
        stp_inits = np.vstack((s_inits, theta_inits, zeta_inits)).T
        return stp_inits, vpar_inits

    def compute_boozer_field(self, x):
        # to save on recomputes
        if np.all(self.x_field == x) and (self.field is not None):
            return self.field, self.bri

        self.surf.x = np.copy(x)
        try:
            self.vmec.run()
        except:
            # VMEC failure!
            return None, None

        # Construct radial interpolant of magnetic field
        bri = BoozerRadialInterpolant(
            self.vmec, order=self.interpolant_degree, mpol=self.bri_mpol, ntor=self.bri_ntor, enforce_vacuum=True
        )

        # Construct 3D interpolation
        nfp = self.vmec.wout.nfp
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
        self.field = field
        self.bri = bri
        self.x_field = np.copy(x)
        return field, bri

    def compute_modB(self, field, bri, ns=32, ntheta=32, nphi=32):
        """
        Compute |B| on a grid in Boozer coordinates.
        """
        stz_grid, _ = self.flux_grid(ns, ntheta, nphi, 1)
        field.set_points(stz_grid)
        modB = field.modB().flatten()
        return modB

    def compute_modB_vmec(self, ns=32, ntheta=32,nphi=32, smin=0.02, smax=1.0):
        """
        Compute |B| on a tensor product grid VMEC coordinates, (s, theta, phi).

        ns, nphi, ntheta: number of samples per dimension.
        smin, smax: min and max values of the normalized toroidal flux. floats in [0,1].

        return a 1d array of evals, length ns*ntheta*nphi
        """
        # TODO: @neil, check this function works

        self.surf.x = np.copy(x)
        # try to run vmec
        try:
            self.vmec.run()
        except:
            # VMEC failure!
            return []

        # use fixed positions
        s = np.linspace(s_min, s_max, ns)
        theta = np.linspace(0, 2*np.pi, ntheta)
        phi = np.linspace(0, 2*np.pi/self.surf.nfp, nphi)

        # potentially run vmec and compute the geometric quantites
        data = vmec_compute_geometry(self.vmec, s, theta, phi) # 3d array

        # return a 1d array
        modB = data.modB.flatten()
        return np.copy(modB)

    def compute_mu(self, field, bri, stz_inits, vpar_inits):
        """
        Compute |B| on a grid in Boozer coordinates.
        """
        field.set_points(stz_inits + np.zeros(np.shape(stz_inits)))
        modB = field.modB().flatten()
        vperp_squared = FUSION_ALPHA_SPEED_SQUARED - vpar_inits**2
        mu = vperp_squared / 2 / modB
        return mu

    def compute_mu_crit(self, field, bri, ns=64, ntheta=64, nphi=64):
        """
        Compute |B| on a grid in Boozer coordinates.
        """
        modB = self.compute_modB(field, bri, ns, ntheta, nphi)
        Bmax = np.max(modB)
        mu_crit = FUSION_ALPHA_SPEED_SQUARED / 2 / Bmax
        return mu_crit

    # set up the objective
    def compute_confinement_times(self, x, stz_inits, vpar_inits, tmax, field=None, bri=None):
        """
        Trace particles in boozer coordinates according to the vacuum GC
        approximation using simsopt.

        x: a point describing the current vmec boundary
        stz_inits: (n,3) array of (s,theta,zeta) points
        vpar_inits: (n,) array of vpar values
        tmax: max tracing time
        """
        n_particles = len(vpar_inits)

        if field is None:
            field, bri = self.compute_boozer_field(x)
        if field is None:
            # VMEC failure
            return -np.inf * np.ones(len(stz_inits))

        stopping_criteria = [MaxToroidalFluxStoppingCriterion(0.99), MinToroidalFluxStoppingCriterion(0.01)]

        # comm = MPI.COMM_WORLD

        # trace
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
            return -np.inf * np.ones(len(stz_inits))

        exit_times = np.zeros(n_particles)
        for ii, res in enumerate(res_zeta_hits):

            # check if particle hit stopping criteria
            if len(res) > 0:
                if int(res[0, 1]) == -1:
                    # particle hit MaxToroidalFluxCriterion
                    exit_times[ii] = res[0, 0]
                if int(res[0, 1]) == -2:
                    # particle hit MinToroidalFluxCriterion
                    exit_times[ii] = tmax
            else:
                # didnt hit any stopping criteria
                exit_times[ii] = tmax

        return exit_times


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    vmec_input = "../vmec_input_files/input.nfp4_QH_warm_start_high_res"
    max_mode = 1
    aspect_target = 7.0
    major_radius = 1.7 * aspect_target
    target_volavgB = 1.0

    tracer = TraceBoozer(
        vmec_input,
        n_partitions=1,
        max_mode=max_mode,
        major_radius=major_radius,
        aspect_target=aspect_target,
        target_volavgB=target_volavgB,
        tracing_tol=1e-8,
        interpolant_degree=3,
        interpolant_level=8,
        bri_mpol=8,
        bri_ntor=8,
    )
    x0 = tracer.x0

    tmax = 1e-4
    n_particles = 10

    # tracing points
    s_label = 0.25
    stz_inits, vpar_inits = tracer.sample_surface(n_particles, s_label)
    c_times = tracer.compute_confinement_times(x0, stz_inits, vpar_inits, tmax)
    feat = 3.5 * np.exp(-2 * c_times / tmax)
    if rank == 0:
        print("")
        print("energy", np.mean(feat))
        print("loss frac", np.mean(c_times < tmax))