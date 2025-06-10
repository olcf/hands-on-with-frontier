# Python: Galaxy Evolution

Scientific simulations generate large amounts of data on Frontier (about 100 Terabytes per day for some applications).
Because of how large some datafiles may be, it is important that writing and reading these files is done as fast as possible.
Less time spent doing input/output (I/O) leaves more time for advancing a simulation or analyzing data.

One of the most utilized file types is the Hierarchical Data Format (HDF), specifically the HDF5 format.
[HDF5](https://www.hdfgroup.org/solutions/hdf5/) is designed to manage large amounts of data and is built for fast I/O processing and storage.
An HDF5 file is a container for two kinds of objects: "datasets", which are array-like collections of data, and "groups", which are folder-like containers that hold datasets and other groups.

There are various tools that allow users to interact with HDF5 data, but we will be focusing on [h5py](https://docs.h5py.org/en/stable/) -- a Python interface to the HDF5 library.
h5py provides a simple interface to exploring and manipulating HDF5 data as if they were Python dictionaries or NumPy arrays.
For example, you can extract specific variables through slicing, manipulate the shapes of datasets, and even write completely new datasets from external NumPy arrays.

Both HDF5 and h5py can be compiled with MPI support, which allows you to optimize your HDF5 I/O in parallel.
MPI support in Python is accomplished through the [mpi4py](https://mpi4py.readthedocs.io/en/stable/) package, which provides complete Python bindings for MPI.
Building h5py against mpi4py allows you to write to an HDF5 file using multiple parallel processes, which can be helpful for users handling large datasets in Python.
h5Py is available after loading the default Python module on Frontier, but it has not been built with parallel support.

This hands-on challenge will teach you how to build a personal, parallel-enabled version of h5py and how to write an HDF5 file in parallel using mpi4py and h5py.
Afterwards, using mpi4py and h5py, you will fix a galaxy evolution script that showcases the effects of tidal stripping for colliding galaxies.

Our plan for building parallel h5py is to:

* Create a new virtual environment using conda
* Install mpi4py from source
* Install h5py from source
* Test our build with a Python script

After successfully testing your build, you will then have the opportunity to complete a challenge involving the simulation of two galaxies colliding and how to get the simulation working with parallel h5py.

## Setting up the environment

Building h5py from source is highly sensitive to the current environment variables set in your profile.
Because of this, it is extremely important that all the modules and conda environments we plan to load are done in the correct order, so that all the environment variables are set correctly.
First, we will unload all the current modules that you may have previously loaded on Frontier and then immediately load the default modules.
Assuming you cloned the repository in your home directory:

```bash
$ cd ~/hands-on-with-frontier/challenges/Python_Galaxy_Evolution
$ source ~/hands-on-with-frontier/misc_scripts/deactivate_envs.sh
$ module reset
```

The `source deactivate_envs.sh` command is only necessary if you already have existing conda environments active.
The script unloads all of your previously activated conda environments, and no harm will come from executing the script if that does not apply to you.

Next, we will load the gnu compiler module (most Python packages assume GCC), hdf5 module (necessary for h5py):

```bash
$ module load PrgEnv-gnu/8.5.0 
$ module load cray-hdf5-parallel/1.12.2.9
$ module load miniforge3
```

We are in a "base" conda environment, but we need to activate a pre-built conda environment that has `mpi4py` and `h5py`.
Due to the specific nature of conda on Frontier, we will be using `source activate` instead of `conda activate` to activate our environment:

```bash
$ source activate /lustre/orion/world-shared/stf007/msandov1/crash_course_envs/h5pympi-frontier
```

The path to the environment should now be displayed in "( )" at the beginning of your terminal lines, which indicate that you are currently using that specific conda environment. 
If you check with `which python3`, you should see that you're properly in the new environment:

```bash
$ which python3
/lustre/orion/world-shared/stf007/msandov1/crash_course_envs/h5pympi-frontier/bin/python3
```

## Testing parallel h5py

We will test our build by trying to write an HDF5 file in parallel using 42 MPI tasks.

First, change directories to your scratch area and copy over the python and batch scripts:

```bash
$ cd /lustre/orion/PROJECT_ID/scratch/${USER}/
$ mkdir h5py_test
$ cd h5py_test
$ cp ~/hands-on-with-frontier/challenges/Python_Galaxy_Evolution/hello_mpi.py .
$ cp ~/hands-on-with-frontier/challenges/Python_Galaxy_Evolution/hdf5_parallel.py .
$ cp ~/hands-on-with-frontier/challenges/Python_Galaxy_Evolution/submit_hello.sbatch .
$ cp ~/hands-on-with-frontier/challenges/Python_Galaxy_Evolution/submit_h5py.sbatch .
```

Let's test that mpi4py is working properly first by executing the example Python script "hello_mpi.py".
To do so, we will be submitting a job to the batch queue with "submit_hello.sbatch":

```bash
$ sbatch --export=NONE submit_hello.sbatch
```

Once the batch job makes its way through the queue, it will run the "hello_mpi.py" script with 42 MPI tasks.
If mpi4py is working properly, in `mpi4py-<JOB_ID>.out` you should see output similar to:

```
Hello from MPI rank 21 !
Hello from MPI rank 23 !
Hello from MPI rank 28 !
Hello from MPI rank 40 !
Hello from MPI rank 0 !
Hello from MPI rank 1 !
Hello from MPI rank 32 !
.
.
.
```

If you see this, great, it means that mpi4py was built successfully in your environment.

Finally, let's see if we can get these tasks to write to an HDF5 file in parallel using the "hdf5_parallel.py" script:

```python
# hdf5_parallel.py
from mpi4py import MPI
import h5py

comm = MPI.COMM_WORLD      # Use the world communicator
mpi_rank = comm.Get_rank() # The process ID (integer 0-41 for a 42-process job)
mpi_size = comm.Get_size() # Total amount of ranks

with h5py.File('output.h5', 'w', driver='mpio', comm=MPI.COMM_WORLD) as f:
    dset = f.create_dataset('test', (42,), dtype='i')
    dset[mpi_rank] = mpi_rank

comm.Barrier()

if (mpi_rank == 0):
    print('42 MPI ranks have finished writing!')
```

The MPI tasks are going to write to a file named "output.h5", which contains a dataset called "test" that is of size 42 (assigned to the "dset" variable in Python).
Each MPI task is going to assign their rank value to the "dset" array in Python, so we should end up with a dataset that contains 0-41 in ascending order.

Time to execute "hdf5_parallel.py" by submitting "submit_h5py.sbatch" to the batch queue:

```bash
$ sbatch --export=NONE submit_h5py.sbatch
```

Provided there are no errors, you should see "42 MPI ranks have finished writing!" in the `h5py.<JOB_ID>.out` output file, and there should be a new file called "output.h5" in your directory.
To see explicitly that the MPI tasks did their job, you can use the `h5dump` command to view the dataset named "test" in output.h5:

```
$ h5dump output.h5

HDF5 "output.h5" {
GROUP "/" {
   DATASET "test" {
      DATATYPE  H5T_STD_I32LE
      DATASPACE  SIMPLE { ( 42 ) / ( 42 ) }
      DATA {
      (0): 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
      (19): 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
      (35): 35, 36, 37, 38, 39, 40, 41
      }
   }
}
}
```

If you see the above output, then congratulations you have used one of the fastest computers in the world to write a parallel HDF5 file in Python!

## Galaxy Collision Challenge

Now for the EXTRA fun part, using mpi4py and h5py together to simulate and generate scientific data (or *partially* scientific, at least).
You will be using your newfound h5py and mpi4py knowledge to simulate two galaxies colliding.
The results of the simulation will look something like this:

<p align="center" width="100%">
    <img width="50%" src="images/galaxy_collision.gif">
</p>

First, similar to before, change directories to your Lustre scratch area and copy over the python and batch scripts:

```bash
$ cd /lustre/orion/PROJECT_ID/scratch/${USER}/
$ mkdir galaxy_challenge
$ cd galaxy_challenge
$ cp ~/hands-on-with-frontier/challenges/Python_Galaxy_Evolution/galaxy.py .
$ cp ~/hands-on-with-frontier/challenges/Python_Galaxy_Evolution/generate_animation.py .
$ cp ~/hands-on-with-frontier/challenges/Python_Galaxy_Evolution/submit_galaxy.sbatch .
```

The two scripts of interest are called `galaxy.py` and `generate_animation.py`.
The `galaxy.py` script generates an HDF5 file using mpi4py and h5py, while `generate_animation.py` just creates a GIF of the results.
You will be dealing with `galaxy.py`.

The goal of `galaxy.py` is to simulate an infalling galaxy made up of "particles" (stars) and a "nucleus" (the compact central region) colliding with a bigger host galaxy.
This would require a lot of code for it to be the most accurate ("many body" problems in physics are complicated); however, we made some physical assumptions to simplify the problem so that it is less complicated but still results in a roughly accurate galactic event.
Even with simplifying things down, this script does not run quickly when not using MPI, as the amount of stars you want to simulate over a given time period quickly slows things down.
We will be simulating 1000 stars and it takes about 2 minutes for the script to complete on Frontier when only using 1 MPI task, while completing in about 20 seconds when using 8 MPI tasks.

In this challenge, you will be using 8 MPI tasks to help speed up the computations by splitting up the particles across your MPI tasks (each MPI task will only simulate a subset of the total number of particles).
The tasks will then write their subset of the data in parallel to an HDF5 file that will hold the entire final dataset.

Luckily all the physics related stuff is done for you and all you have to worry about is changing a few h5py lines for the code to perform properly.
Specifically, there are five lines that need fixing in the `galaxy.py` script (marked by the "TO-DO" comments on lines 207-211):

```python
 # Create dummy data with correct shape
dummy_data = np.empty((N_part, t_size, 3))*0. # Shape: [Number of particles, Number of timesteps, Number of dimensions (x, y, z)]
dummy_nuc  = np.empty((1, t_size, 3))*0.

 # Open and initialize HDF5 file with dummy data
f = h5py.File('galaxy.hdf5', 'w', driver= , comm= )# TO-DO
dset_pos_pt  = f.create_dataset("pos_pt",  data=  )# TO-DO
dset_vel_pt  = f.create_dataset("vel_pt",  data=  )# TO-DO
dset_pos_nuc = f.create_dataset("pos_nuc", data=  )# TO-DO
dset_vel_nuc = f.create_dataset("vel_nuc", data=  )# TO-DO
```

Your challenge is to: 

1. Supply the necessary arguments to get h5py to work with mpi4py (the first TO-DO line), and
2. Supply the necessary "dummy data" variables (either `dummy_data` or `dummy_nuc`) so that the shapes of the HDF5 datasets are correct (the rest of the TO-DO lines).

A major question to help you: "What arguments were used when testing the `hdf5_parallel.py` script earlier?"
If you're having trouble, you can check `galaxy_solution.py` in the `solution` directory.
Although you only have to deal with a small section of `galaxy.py` to complete the challenge, feel free to explore the entire script and see what the rest of it is doing.

To do this challenge:

0. Copy over the scripts into your member work directory (as described further above in this section).
1. Determine the missing pieces of the five "TO-DO" lines.
2. Use your favorite editor to enter the missing pieces into `galaxy.py`. For example:

    ```bash
    $ vi galaxy.py
    ```

3. Submit a job:

    ```bash
    $ sbatch --export=NONE submit_galaxy.sbatch
    ```

4. If you fixed the script, you should see something similar to the output below in `galaxy-<JOB_ID>.out` after the job completes:

    ```python
    MPI Rank 3 : Simulating my particles took 16.70100736618042 s
    MPI Rank 0 : Simulating my particles took 16.869715690612793 s
    MPI Rank 1 : Simulating my particles took 17.179004669189453 s
    MPI Rank 6 : Simulating my particles took 17.414162635803223 s
    MPI Rank 4 : Simulating my particles took 17.449076890945435 s
    MPI Rank 2 : Simulating my particles took 17.516165733337402 s
    MPI Rank 5 : Simulating my particles took 18.588200092315674 s
    MPI Rank 7 : Simulating my particles took 19.119014263153076 s
    Success!
    ```

If you got the script to successfully run, then congratulations!

After you complete the challenge, you can run `generate_anmation.py` (in the same directory you ran your simulation) to generate your personal `galaxy_collision.gif` based on your simulation:

```bash
$ python3 generate_animation.py
```

This will take around a minute to complete, but will result in your own GIF.
You can then transfer this GIF to your computer with Globus, `scp`, or `sftp` to keep as a "souvenir" from the challenge.

## Environment Information

> WARNING: This is NOT part of the challenge, but just context for how the mpi4py+h5py environment we used was installed

Here's how the h5py+mpi4py environment was built:

```bash
$ module load PrgEnv-gnu/8.5.0 
$ module load cray-hdf5-parallel/1.12.2.9
$ module load miniforge3

$ conda create -p /lustre/orion/world-shared/stf007/msandov1/crash_course_envs/h5pympi-frontier python=3.10 numpy scipy matplotlib -c conda-forge

$ source activate /lustre/orion/world-shared/stf007/msandov1/crash_course_envs/h5pympi-frontier

$ MPICC="cc -shared" pip install --no-cache-dir --no-binary=mpi4py mpi4py

$ HDF5_MPI="ON" CC=cc HDF5_DIR=${HDF5_ROOT} pip install --no-cache-dir --no-binary=h5py h5py
```

## Additional Resources

* [h5py Documentation](https://docs.h5py.org/en/stable/)
* [mpi4py Documentation](https://mpi4py.readthedocs.io/en/stable/)
* [HDF5 Support Page](https://portal.hdfgroup.org/display/HDF5/HDF5)
