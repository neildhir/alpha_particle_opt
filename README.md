# Derivative-free Optimisation of Fast-Ion Confinement in Stellarators

Confining energetic ions such as alpha particles is a crucial consideration in the design of stellarators. While direct measurement of alpha confinement through numerical simulation of guiding-centre trajectories has been deemed computationally expensive and noisy, proxy metrics – simplified measures of confinement – have been widely employed in the design process due to their computational tractability and proven effectiveness. However, the extent to which these proxies compromise the design optimality remains unclear when compared to relying on direct trajectory calculations. In this study, we employ Bayesian optimisation (BO) to explore stellarator designs for improved alpha particle confinement without resorting to proxy metrics. Specifically, we leverage BO to numerically optimise an objective function that measures alpha particle losses by simulating alpha particle trajectories. Despite the computational overhead associated with this approach, we demonstrate that BO can successfully generate configurations with low alpha particle losses, circumventing the need for proxy metrics and potentially yielding superior designs.

## Usage

Navigate to `src/` then run `python run_one_replicate.py` to run the optimisation.

## Misc

Set `OMP_NUM_THREADS=1` to ensure that MPI is not slowed down by OpenMP.

## Installation instructions for running on the G2 cluster
1. Make a conda env 
  ```
  conda create -n "particle_tracing_bo" python=3.8.0 ipython
  ```
  And activate it,
  ```
  conda activate particle_tracing_bo
  ```

2. Install VMEC. First clone the VMEC2000 repo. I cloned VMEC on VMEC 0.0.5 on August 11, 2024.
    ```
    cd ~
    git clone git@github.com:hiddenSymmetries/VMEC2000.git
    cd VMEC2000
    ```
    Now install the necessary packages.
    ```
    conda install numpy
    conda install cmake scikit-build ninja f90wrap
    ```
    I also installed scalapack because I couldn't find it on G2 at the time. I don't remember how I did this, but I am sure David knows how to do this. In hindsight, I think it exists inside `/lib/x86_64-linux-gnu/`. I added the following line to my `~/.bashrc` to set the environment variable for scalapack,
    ```
    export LD_LIBRARY_PATH=/home/map454/scalapack/usr/local/lib
    ```
    Now edit the `VMEC2000/cmake_config_file.json` file to look like,
    ```
    {
    "cmake_args": [
           "-DCMAKE_C_COMPILER=mpicc",
           "-DCMAKE_Fortran_COMPILER=mpif90",
           "-DNETCDF_INC_PATH=/usr/include",
           "-DNETCDF_LIB_PATH=/usr/lib/x86_64-linux-gnu",
           "-DSCALAPACK_PREFIX=/home/map454/scalapack/usr/local"]
    }
    ```
    Finally do a `pip install`. 
    ```
    python -m pip install . -v
    ```
    If the install fails, remove the `_skbuild` directory. Install whatever you need or adjust the `cmake_config_file.json` and try again.

3. Install simsopt with any necessary MPI related modules, such as mpi4py. We are going to be using the `general_constrained_optimization` branch of my fork.
    ```
    git clone git@github.com:mishapadidar/simsopt.git
    pip install --user -e .[MPI]
    git checkout general_constrained_optimization
    ```

4. Now clone the directory for the Bayesian optimization for particle losses,
    ```
    git clone git@github.com:neildhir/alpha_particle_opt.git
    git checkout bo_branch
    ```

5. Add the repo to your `PYTHONPATH` so that the directory structure is recognized by python. Modify your `.bashrc` to contain the following line (replacing the path with the relevant one)`,
    ```
    export PYTHONPATH="/home/map454/neil_dhir/alpha_particle_opt"
    ```
    You can always delete this line if you want to delete the repo.

6. Set the following environment variable to ensure that MPI is not slowed down by OpenMP. Add the following command to your `~/.bashrc` or just run, 
    ```
    export OMP_NUM_THREADS=1
    ```
