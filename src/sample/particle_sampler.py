import numpy as np
from src.sample.radial_density import RadialDensity
from src.sample.angle_density import compute_det_jac_dcart_dbooz
from src.utils.constants import V_MAX  
from simsopt._core import Optimizable

class NonUniformSampler:

    def __init__(self, mpi, nfp, s_label=0.25, n_particles = 128):
        """
        A class for randomly sampling or positioning particles in flux coordinates.
        The sampling is not necessarily uniform over surfaces.

        mpi: instance of MpiPartition
        nfp: number of field periods.
        s_label: str, the normalized toroidal flux (0, 1) for the surface that will
            be sampled by the sample_surface function.
        n_particles: int, number of particles to sample.
        """

        self.mpi = mpi
        self.s_label = s_label
        self.nfp = nfp
        self.n_particles = n_particles


    def sample_surface(self):
        """
        Nonuniformly sample a flux surface with normalized toroidal flux s_label. 
        The toroidal and poloidal values are sampled uniformly on [0, 2pi/nfp] x [0, 2pi].
        The parallel velocity is sampled uniformly over [-V_MAX, V_MAX].

        return stz_inits, vpar_inits
        stz_inits: (n_particles, 3) array of points (s, theta, zeta) on the flux surface.
        vpar_inits: (n_particles,) array of parallel velocity values
        """
        n_particles = self.n_particles

        # sampling over (theta,zeta,vpar) for a fixed surface
        s_inits = self.s_label*np.ones(n_particles)
        theta_inits = np.zeros(n_particles)
        zeta_inits = np.zeros(n_particles)
        vpar_inits = np.zeros(n_particles)

        if self.mpi.proc0_groups:
            # randomly sample theta,zeta,vpar
            theta_inits = np.random.uniform(0, 2 * np.pi, n_particles)
            zeta_inits = np.random.uniform(0, 2 * np.pi / self.nfp, n_particles)
            vpar_inits = np.random.uniform(-V_MAX, V_MAX, n_particles)
        
        # broadcast the points
        self.mpi.comm_groups.Bcast(theta_inits, root=0)
        self.mpi.comm_groups.Bcast(zeta_inits, root=0)
        self.mpi.comm_groups.Bcast(vpar_inits, root=0)

        # stack the samples
        stz_inits = np.vstack((s_inits, theta_inits, zeta_inits)).T
        return stz_inits, vpar_inits

    def sample_volume(self):
        """
        Sample the stellarator volume. 
        The angles and the parallel velocity are sampled by sample_surface.
        The normalized toroidal flux is sampled using the radial density sampler.
        The parallel velocity is sampled uniformly over [-V_MAX, V_MAX].

        return stz_inits, vpar_inits
        stz_inits: (n_particles, 3) array of points (s, theta, zeta).
        vpar_inits: (n_particles,) array of parallel velocity values
        """            
        # randomly sample the angles and parallel velocity
        stz_inits, vpar_inits = self.sample_surface()
        n_particles = len(vpar_inits)

        # sample s
        s_inits = np.zeros(n_particles)
        if self.mpi.proc0_groups:
            sampler = RadialDensity()
            s_inits = sampler.sample(n_particles)

        self.mpi.comm_groups.Bcast(s_inits, root=0)

        stz_inits[:,0] = s_inits.flatten()

        return stz_inits, vpar_inits


class UniformSampler(Optimizable):

    def __init__(self, booz, mpi, nfp, s_label=0.25, n_particles = 128,
                 ntheta=32, nzeta=32):
        """
        A class for (approximately) uniformly sampling particle positions on
        a flux surface. 

        booz: instance of Booz class,
        mpi: instance of MpiPartition,
        nfp: number of field periods,
        s_label: str, the normalized toroidal flux (0, 1) for the surface that will
            be sampled by the sample_surface function.
        n_particles: int, number of particles to sample.
        ntheta, nzeta: int, number of points to evaluate the jacobian at.
        """

        self.nfp = nfp
        self.mpi = mpi
        self.s_label = s_label
        self.n_particles = n_particles
        self.ntheta = ntheta
        self.zeta = nzeta

        # form the grid
        theta1d = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
        zeta1d = np.linspace(0, 2 * np.pi / nfp, nzeta, endpoint=False)
        [thetas, zetas] = np.meshgrid(theta1d, zeta1d)
        stz_grid = np.zeros((ntheta * nzeta, 3))
        stz_grid[:, 0] = s_label
        stz_grid[:, 1] = thetas.flatten()
        stz_grid[:, 2] = zetas.flatten()
        self.stz_grid = stz_grid
        self.theta1d = theta1d
        self.zeta1d = zeta1d

        # VMEC failures
        self.fail_value = np.zeros(ntheta*nzeta)

        # for caching
        self.need_to_run_code = True

        self.booz = booz
        super().__init__(depends_on=[booz])

    def recompute_bell(self, parent=None):
        """
        This function will get called any time any of the DOFs of the
        parent class (vmec) change.
        """
        self.need_to_run_code = True

    def cache_results(self, jac, is_success):
        """
        Cache the tracing results.
        """
        self.jac = jac
        self.is_success = is_success
        self.need_to_run_code = False

    def compute(self):
        """
        Compute the absolute value determinant jacobian of the transformation
        from Boozer coordinates to Cartesian:
            |D(X,Y,Z)/D(s,theta,zeta)|

        return: jac, is_success
            jac: array of length ntheta*nzeta. Is populated with fail_value is 
            the computation (VMEC) fails. Otherwise it contains the jacobian
            determinant.
            is_success: bool, True is the computation succeeds.
        """    
        # check cache
        if not self.need_to_run_code:
            return self.jac, self.is_success
        
        field, bri, is_success = self.booz.compute()

        # VMEC failure
        if not is_success:
            self.cache_results(self.fail_value, is_success)
            return self.fail_value, is_success

        jac = compute_det_jac_dcart_dbooz(field, self.stz_grid)

        self.cache_results(jac, is_success)
        return jac, is_success

    def sample_surface(self):
        """
        Sample a surface (approximately) uniformly. 
        The parallel velocity is sampled uniformly over [-V_MAX, V_MAX].

     
        The method used relies on discretizing the Boozer surface into cells where 
        |det(jac)| is approximately constant. When this condition holds, uniformly
        sampling over (theta, zeta) is equivalent to uniformly sampling over space.

        Steps:
            - Evaluate the angle density on a grid of points across the surface.
            - Thinking of the grid as discrete cells, approximate the PMF of each 
                cells as the angle density times the cell area.
            - Now to sample points (approximately) uniformly,
                - Randomly sample a grid cell according to the PMF.
                - uniformly sample the two angles.

        return: stz, vpars 
            stz: (n_particles, 3) array of points (s, theta, zeta) uniformly sampled.
                If the BoozXform failed, return an array of zeros.
            vpars: (n_particles, ) array of parallel velocities uniformly sampled.
                If the BoozXform failed, return an array of zeros.
        """
        n_particles = self.n_particles
        stz = np.zeros((n_particles, 3))
        vpars = np.zeros(n_particles)

        jac, is_success = self.compute()
        if not is_success:
            return stz, vpars

        pmf = jac/np.sum(jac)   

        dtheta = np.diff(self.theta1d)[0]
        dzeta = np.diff(self.zeta1d)[0]

        thetas = np.zeros(n_particles)
        zetas = np.zeros(n_particles)
        if self.mpi.proc0_groups:
            # sample cells according to the pmf
            idxs = np.arange(0, len(jac), dtype=int)
            idxs_sample = np.random.choice(idxs,
                                    size=n_particles, p = pmf)
            cells = self.stz_grid[idxs_sample]

            # sample inside cells
            thetas = np.random.uniform(cells[:,1], cells[:,1] + dtheta)
            zetas = np.random.uniform(cells[:,2], cells[:,2] + dzeta)

            # sample parallel velocity
            vpars = np.random.uniform(-V_MAX, V_MAX, n_particles)

        self.mpi.comm_groups.Bcast(thetas, root=0)
        self.mpi.comm_groups.Bcast(zetas, root=0)
        self.mpi.comm_groups.Bcast(vpars, root=0)


        stz[:,0] = self.s_label
        stz[:,1] = np.copy(thetas)
        stz[:,2] = np.copy(zetas)

        return stz, vpars
    
    def sample_volume(self):
        """
        Sample from the Stellarator volume, uniformly over surfaces, 
        and according to the radial density over the radial direction.
        The parallel velocity is sampled uniformly over [-V_MAX, V_MAX].

        return: stz, vpars 
            stz: (n_particles, 3) array of points (s, theta, zeta) uniformly sampled.
                If the BoozXform failed, return an array of zeros.
            vpars: (n_particles, ) array of parallel velocities uniformly sampled.
                If the BoozXform failed, return an array of zeros.
        """

        stz, vpars = self.sample_surface()

        if not self.is_success:
            return stz, vpars

        # sample s
        s = np.zeros(self.n_particles)
        if self.mpi.proc0_groups:
            sampler = RadialDensity()
            s = sampler.sample(self.n_particles)

        self.mpi.comm_groups.Bcast(s, root=0)

        stz[:,0] = s.flatten()
        return stz, vpars


class FluxSurfaceGrid:

    def __init__(self,
                 nfp,
                 s_label=0.25,
                 ntheta=32, 
                 nzeta=32, 
                 nvpar=32):
        """
        A class for computing tensor product grids on flux surfaces.

        nfp: number of field periods.
        s_label: float or str, either the normalized toroidal flux (0, 1) or the 
            string 'volume' denoting the volume it is sampled in.
        ntheta, nzeta, nvpar: number of points taken per direction, if a grid is used.

        """
        self.s_label = s_label
        self.nfp = nfp
        self.ntheta = ntheta
        self.nzeta = nzeta
        self.nvpar = nvpar

    def surface_grid(self):
        """
        Build a tensor product grid on a single surface in flux coordinates. The grid is
        uniformly spaced over the two angles and parallel velocity.

        return stz_inits, vpar_inits
        stz_inits: (ntheta*nzeta*nvpar, 3) array of points (s, theta, zeta) on the grid.
        vpar_inits: (n_vpar,) array of parallel velocity values
        """
        theta1d = np.linspace(0, 2 * np.pi, self.ntheta, endpoint=False)
        zeta1d = np.linspace(0, 2 * np.pi / self.nfp, self.nzeta, endpoint=False)
        vpar1d = np.linspace(-V_MAX, V_MAX, self.nvpar)

        # build a mesh
        [thetas, zetas, vpars] = np.meshgrid(theta1d, zeta1d, vpar1d)
        stz_inits = np.zeros((self.ntheta * self.nzeta * self.nvpar, 3))
        stz_inits[:, 0] = self.s_label
        stz_inits[:, 1] = thetas.flatten()
        stz_inits[:, 2] = zetas.flatten()
        vpar_inits = vpars.flatten()
        return stz_inits, vpar_inits
    

# if __name__ == "__main__":
#     sampler = ParticleSampler(nfp=4, s_label=0.25, n_particles=2)
#     print(sampler.sample_surface())
#     print(sampler.sample_volume())

#     sampler = FluxSurfaceGrid(nfp=4, s_label=0.3, ntheta=1, nzeta=3, nvpar=2)
#     print(sampler.surface_grid())
