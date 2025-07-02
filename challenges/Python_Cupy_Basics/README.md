# Python: CuPy Basics

GPU computing has become a big part of the data science landscape, as array operations with NVIDIA GPUs can provide considerable speedups over CPU computing.
Although GPU computing on Frontier is often utilized in codes that are written in Fortran and C, GPU-related Python packages are quickly becoming popular in the data science community.
One of these packages is [CuPy](https://cupy.dev/), a NumPy/SciPy-compatible array library accelerated with NVIDIA CUDA.

CuPy is a library that implements NumPy arrays on NVIDIA GPUs by utilizing CUDA Toolkit libraries like cuBLAS, cuRAND, cuSOLVER, cuSPARSE, cuFFT, cuDNN and NCCL.
Although optimized NumPy is a significant step up from Python in terms of speed, performance is still limited by the CPU (especially at larger data sizes) -- this is where CuPy comes in.
Because CuPy's interface is nearly a mirror of NumPy, it acts as a replacement to run existing NumPy/SciPy code on NVIDIA CUDA platforms, which helps speed up calculations further.
CuPy supports most of the array operations that NumPy provides, including array indexing, math, and transformations.
Most operations provide an immediate speed-up out of the box, and some operations are sped up by over a factor of 100 (see CuPy benchmark timings below, from the [Single-GPU CuPy Speedups](https://medium.com/rapids-ai/single-gpu-cupy-speedups-ea99cbbb0cbb) article).

<p align="center" width="100%">
    <img width="50%" src="images/cupy_chart.png">
</p>

Compute nodes equipped with NVIDIA GPUs will be able to take full advantage of CuPy’s capabilities on the system, providing significant speedups over NumPy-written code. CuPy with AMD GPUs is still being explored, and the same performance is not guaranteed (especially with larger data sizes).  

Instructions for Odo are available in this guide, but users must note that the CuPy developers have labeled this method as experimental and has limitations.

&nbsp;

In this challenge, you will:

* Learn how to install CuPy into a custom conda environment
* Learn the basics of CuPy
* Apply what you've learned in a debugging challenge

&nbsp;

## Setting up our environment

>>  ---
> Before setting up your environment, you must exit and log back in so that you have a fresh login shell. This is to ensure that no previously activated environments exist in your $PATH environment variable. Additionally, you should execute module reset.
>>  ---

First, we will unload all the current modules that you may have previously loaded on Odo and then immediately load the default modules.
Assuming you cloned the repository in your home directory:

```bash
$ cd ~/hands-on-with-odo/challenges/Python_Cupy_Basics
$ source ~/hands-on-with-odo/misc_scripts/deactivate_envs.sh
$ module reset
```

The `source deactivate_envs.sh` command is only necessary if you already have existing conda environments active.
The script unloads all of your previously activated conda environments, and no harm will come from executing the script if that does not apply to you.

Next, we will load the gnu compiler module (most Python packages assume GCC), relevant GPU module (necessary for CuPy):

```bash
$ module load PrgEnv-gnu/8.6.0 
$ module load rocm/5.7.1
$ module load craype-accel-amd-gfx90a
$ module load miniforge3 
```

We loaded the "base" conda environment, but we need to activate a pre-built conda environment that has CuPy.
Due to the specific nature of conda on Odo, we will be using `source activate` instead of `conda activate` to activate our new environment:

```bash
$ source activate /gpfs/wolf2/olcf/stf007/world-shared/9b8/crashcourse_envs/cupy-odo
```

The path to the environment should now be displayed in "( )" at the beginning of your terminal lines, which indicates that you are currently using that specific conda environment.
If you check with `which python3`, you should see that you're properly in the new environment:

```bash
$ which python3
/gpfs/wolf2/olcf/stf007/world-shared/9b8/crashcourse_envs/cupy-odo/bin/python3
```


&nbsp;

## Getting Started With CuPy

>> ---
> NOTE: Assuming you are continuing from the previous section, you do not need to load any modules.
> However, if you logged out after finishing the previous section, you must load the modules followed by activating your CuPy conda environment before moving on.
>> ---

Before we start testing the CuPy scripts provided in this repository, let's go over some of the basics.
The developers provide a great introduction to using CuPy in their user guide under the [CuPy Basics](https://docs.cupy.dev/en/stable/user_guide/basic.html) section.
We will be following this walkthrough on Odo.
This is done to illustrate the basics, but participants should **NOT** explicitly follow along (as resources are limited on Odo and interactive jobs will clog up the queue).

The syntax below assumes being in a Python shell with access to 4 GPUs; however, Odo interactive nodes have 8 GPUs allocated to CuPy by default. 

As is the standard with NumPy being imported as "np", CuPy is often imported in a similar fashion:

```python
>>> import numpy as np
>>> import cupy as cp
```

Similar to NumPy arrays, CuPy arrays can be declared with the `cupy.ndarray` class.
NumPy arrays will be created on the CPU (the "host"), while CuPy arrays will be created on the GPU (the "device"):

```python
>>> x_cpu = np.array([1,2,3])
>>> x_gpu = cp.array([1,2,3])
```

Manipulating a CuPy array can also be done in the same way as manipulating NumPy arrays:

```python
>>> x_cpu*2.
array([2., 4., 6.])
>>> x_gpu*2.
array([2., 4., 6.])
>>> l2_cpu = np.linalg.norm(x_cpu)
>>> l2_gpu = cp.linalg.norm(x_gpu)
>>> print(l2_cpu,l2_gpu)
3.7416573867739413 3.7416573867739413
```

Useful functions for initializing arrays like `np.linspace`, `np.arange`, and `np.zeros` also have a CuPy equivalent:

```python
>>> cp.zeros(3)
array([0., 0., 0.])
>>> cp.linspace(0,10,11)
array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])
>>> cp.arange(0,11,1)
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
```

CuPy has a concept of a "current device", which is the current activated GPU device that will operate on an array or where future arrays will be allocated.
Most of the time, if not explicitly declared or switched, the initial default device will be GPU 0.
To find out what device a CuPy array is allocated on, you can call the `cupy.ndarray.device` attribute:

```python
>>> x_gpu.device
<CUDA Device 0>
```
To get a total number of devices that you can access, use the `getDeviceCount` function:

```python
>>> cp.cuda.runtime.getDeviceCount()
4
```

The current device can be switched using `cupy.cuda.Device(<DEVICE_ID>).use()`:

```python
>>> cp.cuda.Device(1).use()
>>> x_gpu_1 = cp.array([1, 2, 3, 4, 5])
>>> x_gpu_1.device
<CUDA Device 1>
```

Similarly, you can temporarily switch to a device using the `with` context:

```python
>>> cp.cuda.Device(0).use()
>>> with cp.cuda.Device(3):
...    x_gpu_3 = cp.array([1, 2, 3, 4, 5])
...
>>> x_gpu_0 = cp.array([1, 2, 3, 4, 5])
>>> x_gpu_0.device
<CUDA Device 0>
>>> x_gpu_3.device
<CUDA Device 3>
```

>> ---
> NOTE: In older versions of CuPy, trying to access an array stored on a different GPU resulted in error.
> Transferring an array to the "correct" device before trying to access it is still highly recommended.
>> ---

Now, transfer `x_gpu_0` to "Device 1".
A CuPy array can be transferred to a specific GPU using the `cupy.asarray()` function while on the specific device:

```python
>>> with cp.cuda.Device(1):
...    cp.asarray(x_gpu_0) * 2  # moves x_gpu_0 to GPU 1
...
array([ 2,  4,  6,  8, 10])
```

A NumPy array on the CPU can also be transferred to a GPU using the same `cupy.asarray()` function:

```python
>>> x_cpu = np.array([1, 1, 1]) # create an array on the CPU
>>> x_gpu = cp.asarray(x_cpu)  # move the CPU array to the current device
>>> x_gpu
array([1, 1, 1])
```

To transfer from a GPU back to the CPU, you use the `cupy.asnumpy()` function instead:

```python
>>> x_gpu = cp.zeros(3)  # create an array on the current device
>>> x_cpu = cp.asnumpy(x_gpu)  # move the GPU array to the CPU
>>> x_cpu
array([ 0., 0., 0.])
```

Associated with the concept of current devices are current "streams".
In CuPy, all CUDA operations are enqueued onto the current stream, and the queued tasks on the same stream will be executed in serial (but asynchronously with respect to the CPU).
This can result in some GPU operations finishing before some CPU operations.
As CuPy streams are out of the scope of this challenge, you can find additional information in the [CuPy User Guide](https://docs.cupy.dev/en/stable/user_guide/index.html).

Congratulations, you now know some of the basics of CuPy!

Now let's apply what you've learned.

&nbsp;

## Data Transfer Debugging Challenge

Before asking for a compute node, let's change into our scratch directory and copy over the relevant files.

```
$ cd /gpfs/wolf2/olcf/PROJECT_ID/scratch/${USER}/
$ mkdir cupy_test
$ cd cupy_test
$ cp ~/hands-on-with-odo/challenges/Python_Cupy_Basics/data_transfer.py .
$ cp ~/hands-on-with-odo/challenges/Python_Cupy_Basics/submit_data.sbatch .
```

When a kernel call is required in CuPy, it compiles a kernel code optimized for the shapes and data types of given arguments, sends it to the GPU device, and executes the kernel. 
Due to this, CuPy runs slower on its initial execution.
This slowdown will be resolved at the second execution because CuPy caches the kernel code sent to GPU device.

Now, it's time to dive into `data_transfer.py`:

```python
# data_transfer.py
import numpy as np
import cupy as cp

# Initialize the array x_gpu_0 on GPU 0
x_gpu_0 = cp.arange(10)
cp.cuda.Stream.null.synchronize() # Waits for GPU 0 to finish
print(x_gpu_0.device, 'done')

# Modify a copy of x_gpu_0 on GPU 1 (must send data to GPU 1)
with cp.cuda.Device(1):
        x_gpu_1 = x_gpu_0 # TO-DO
        x_gpu_1 = x_gpu_1**2
        cp.cuda.Stream.null.synchronize() # Waits for GPU 1 to finish
        print(x_gpu_1.device, 'done')

# Modify a copy of x_gpu_0 on GPU 2 (must send data to GPU 2)
with cp.cuda.Device(2):
        x_gpu_2 = x_gpu_0 # TO-DO
        x_gpu_2 = x_gpu_2**3
        cp.cuda.Stream.null.synchronize() # Waits for GPU 2 to finish
        print(x_gpu_2.device, 'done')

# Sum all arrays on the CPU (must send arrays to the CPU)
x_cpu = x_gpu_0 +  x_gpu_1 + x_gpu_2 # TO-DO
print('Finished computing on the CPU\n')

# Summary of our results
print('Results:')
print(x_gpu_0.device, ':', x_gpu_0)
print(x_gpu_1.device, ':', x_gpu_1)
print(x_gpu_2.device, ':', x_gpu_2)
print('CPU: ', x_cpu)
```

The goal of the above script is to calculate `x + x^2 + x^3` on the CPU after calculating `x^2` and `x^3` on separate GPUs.
To do so, the working script initializes `x` on GPU 0, then make copies on both GPU 1 and GPU 2.
After all of the GPUs finish their calculations, the CPU then computes the final sum of all the arrays.
However, running the above script will result in errors, so it is your mission to figure out how to fix it.
Specifically, there are three lines that need fixing in this script (marked by the "TO-DO" comments).

Your challenge is to apply the necessary function calls on the three "TO-DO" lines to transfer the data between the GPUs and CPUs properly.
Some of the questions to help you: What function do I use to pass arrays to a GPU? What function do I use to pass arrays to a CPU?
Also, you can use the error messages to your advantage (they will give you hints as well).
If you're having trouble, you can check `data_transfer_solution.py` in the `solution` directory.

To do this challenge:

1. Determine the missing functions on the three "TO-DO" lines.
2. Use your favorite editor to enter the missing functions into `data_transfer.py`. For example:

    ```bash
    $ vi data_transfer.py
    ```

3. Submit a job:

    ```bash
    $ sbatch --export=NONE submit_data.sbatch
    ```

4. If you fixed the script, you should see the below output in `cupy_xfer-<JOB_ID>.out` after the job completes:

    ```python
    <CUDA Device 0> done
    <CUDA Device 1> done
    <CUDA Device 2> done
    Finished computing on the CPU

    Results:
    <CUDA Device 0> : [0 1 2 3 4 5 6 7 8 9]
    <CUDA Device 1> : [ 0  1  4  9 16 25 36 49 64 81]
    <CUDA Device 2> : [  0   1   8  27  64 125 216 343 512 729]
    CPU:  [  0   3  14  39  84 155 258 399 584 819]
    ```

If you got the script to successfully run, then congratulations!

## Environment Information

> WARNING: This is NOT part of the challenge, but just context for how the CuPy environment we used was installed

Here's how the CuPy environment was built:

```bash
$ module load PrgEnv-gnu/8.6.0 
$ module load rocm/5.7.1
$ module load craype-accel-amd-gfx90a
$ module load miniforge3 

$ conda create -p /gpfs/wolf2/olcf/stf007/world-shared/9b8/crashcourse_envs/cupy-odo python=3.10 numpy=1.26.4 scipy -c conda-forge

$ source activate /gpfs/wolf2/olcf/stf007/world-shared/9b8/crashcourse_envs/cupy-odo

$ export CUPY_INSTALL_USE_HIP=1
$ export ROCM_HOME=${ROCM_PATH}
$ export HCC_AMDGPU_TARGET=gfx90a

$ CC=gcc CXX=g++ pip install --no-cache-dir --no-binary=cupy cupy==12.3.0
```

## Additional Resources

* [CuPy User Guide](https://docs.cupy.dev/en/stable/user_guide/index.html)
* [CuPy Website](https://cupy.dev/)
* [CuPy API Reference](https://docs.cupy.dev/en/stable/reference/index.html)
