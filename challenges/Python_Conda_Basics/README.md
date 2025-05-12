# Python: Conda Basics

In high-performance computing, [Python](https://www.python.org/) is heavily used to analyze scientific data on the system. 
Various Python installations and scientific packages need to be installed to analyze data for our users. These Python installations can become difficult to manage on an HPC system as the programming environment is complicated.  [Conda](https://conda.io/projects/conda/en/latest/index.html), a package and virtual environment manager from the [Anaconda](https://www.anaconda.com/) distribution, helps alleviate these issues. [Miniforge](https://github.com/conda-forge/miniforge) is an open source version of Miniconda, which is what the OLCF crash course will use to be able to utilize conda environments.

Conda allows users to easily install different versions of binary software packages and any required libraries appropriate for their computing platform.
The versatility of conda allows a user to essentially build their own isolated Python environment, without having to worry about clashing dependencies and other system installations of Python.

This hands-on challenge will introduce a user to installing Conda on Odo, the basic workflow of using conda environments, as well as providing an example of how to create a conda environment that uses a different version of Python than the base environment uses on Odo.

&nbsp;

## Inspecting and setting up the environment

First, we will unload all the current modules that you may have previously loaded on Odo:

```bash
$ cd ~/hands-on-with-odo/challenges/Python_Conda_Basics
$ source ~/hands-on-with-odo/misc_scripts/deactivate_envs.sh
$ module reset
```

Next, we need to load the gnu compiler module (most Python packages assume use of GCC), and the miniforge module (allows us to create conda environments):

```bash
$ module load PrgEnv-gnu/8.6.0
$ module load miniforge3
```

This puts you in the "`base`" conda environment (your base-level install that came with a few packages).
Typical best practice is to not install new things into the `base` environment, but to create new environments instead. 
So, next, we will create a new environment using the `conda create` command:

```bash
$ conda create -p /ccsopen/proj/<YOUR_PROJECT_ID>/${USER}/conda_envs/odo/py39-odo python=3.9
```

The "`-p`" flag specifies the desired path and name of your new virtual environment.
The directory structure is case sensitive, so be sure to insert "<YOUR_PROJECT_ID>" as lowercase.
The `${USER}` environment variable is a shortcut for accessing your username.
Directories will be created if they do not exist already (provided you have write-access in that location).
Instead, one can solely use the `--name <your_env_name>` flag which will automatically use your `$HOME` directory.

>>  ---
> NOTE: It is highly recommended to create new environments in the "Project Home" directory (on Odo, this is `/ccsopen/proj/<YOUR_PROJECT_ID>/<YOUR_USER_ID>`).
> This space avoids purges and allows for potential collaboration within your project.
> It is also recommended, for convenience, that you use environment names that indicate the hostname, as virtual environments created on one system will not necessarily work on others.
>>  ---

After executing the `conda create` command, you will be prompted to install "the following NEW packages" -- type "y" then hit Enter/Return.
Downloads of the fresh packages will start and eventually you should see something similar to:

```
Preparing transaction: done
Verifying transaction: done
Executing transaction: done
#
# To activate this environment, use
#
#     $ conda activate /ccsopen/proj/<YOUR_PROJECT_ID>/<YOUR_USER_ID>/conda_envs/odo/py39-odo
#
# To deactivate an active environment, use
#
#     $ conda deactivate
```

Due to the specific nature of conda on Odo, we will be using `source activate` and `source deactivate` instead of `conda activate` and `conda deactivate`.
Let's activate our new environment:

```bash
$ source activate /ccsopen/proj/<YOUR_PROJECT_ID>/${USER}/conda_envs/odo/py39-odo
```

The path to the environment should now be displayed in "( )" at the beginning of your terminal lines, which indicate that you are currently using that specific conda environment.
And if you check with `conda env list` again, you should see that the `*` marker has moved to your newly activated environment:

```
$ conda env list

# conda environments:
#
                      *  /ccsopen/proj/<YOUR_PROJECT_ID>/<YOUR_USER_ID>/conda_envs/odo/py39-odo
base                     /autofs/nccs-svm1_sw/odo/miniforge3/23.11.0
```

&nbsp;

## Installing packages

Next, let's install a package ([NumPy](https://numpy.org/)). 
There are a few different approaches.

One way to install packages into your conda environment is to use [pip](https://pip.pypa.io/en/stable/).
Although pip can install pre-compiled binaries like conda, it can also be used to build packages from source.
This approach is useful if a specific package or package version is not available in the conda repository, or if the pre-compiled binaries don't work on the HPC resources (which is common).
Pip is available to use after installing Python into your conda environment, which we have already done.

>>  ---
> NOTE: Because issues can arise when using conda and pip together (see link in [Additional Resources Section](#refs)), it is recommended to do this only if absolutely necessary. However, as long as you are careful about it, things will probably be fine.
>>  ---

Building from source means you need to take care of some of the dependencies yourself, especially for optimization.
In Odo's case, this means we need to load the `openblas` module.
To build a package from source, use `pip install --no-binary=<package_name> <package_name>`:

```bash
$ module load openblas
$ CC=gcc CXX=g++ pip install --no-binary=numpy numpy --no-cache-dir
```

The `CC=gcc` flag will ensure that we are using the proper compiler and wrapper.
Building from source results in a longer installation time for packages, so you may need to wait a few minutes for the install to finish.
The `no-cache-dir` flag makes sure that no previously built packages that may exist in your cache are used.

After it is finished building, you should see something similar to:

```
Successfully built numpy
Installing collected packages: numpy
Successfully installed numpy-2.0.2
```

Congratulations, you have built NumPy from source in your conda environment!  

Now, let's install using a different method instead, but first we must uninstall the pip-installed NumPy:

```bash
$ pip uninstall numpy
$ module unload openblas
```

The traditional, and more basic, approach to installing/uninstalling packages into a conda environment is to use the commands `conda install` and `conda remove`.
In "regular" Andaconda, installing packages with this method checks the [Anaconda Distribution Repository](https://docs.anaconda.com/anaconda/packages/pkg-docs/) for pre-built binary packages to install.
However, because we are using Miniforge, installing packages checks the [Conda-forge](https://anaconda.org/conda-forge) repository ("channel") instead.
Let's do this to install NumPy:

```bash
$ conda install numpy -c conda-forge
```

Conda handles dependencies when installing pre-built binaries, so  it will automatically install all of the packages NumPy needs for optimization.   

Congratulations, you have just installed NumPy, now let's test it!

&nbsp;

## Testing your new environment

Let's run a small script to test that things installed properly.
Since we are running a small test, we can do this without having to run on a compute node. 

>>  ---
> NOTE: Remember, at larger scales both your performance and your fellow users' performance will suffer if you do not run on the compute nodes.
>>  ---

It is always highly recommended to run on the compute nodes (through the use of a batch job or interactive batch job).

Make sure you're in the correct directory and execute the example Python script:

```
$ cd ~/hands-on-with-odo/challenges/Python_Conda_Basics/
$ python3 hello.py

Hello from Python 3.9.22!
You are using NumPy 2.0.2
```

Congratulations, you have just created your own Python environment and ran on one of the fastest computers in the world!

>>  ---
> Note: If you're doing this challenge for the certificate, you can submit your Python environment for completion. See "Exporting (sharing) an environment" tip below of how to export your environment to a file.
>>  ---

&nbsp;

## Additional Tips

* Cloning an environment:

    It is not recommended to try to install new packages into the base environment.
    Instead, you can clone the base environment for yourself and install packages into the clone.
    To clone an environment, you must use the `--clone <env_to_clone>` flag when creating a new conda environment.
    An example for cloning the base environment into your `$HOME` directory on Odo is provided below:

    ```bash
    $ conda create -p /ccsopen/home/${USER}/.conda/envs/baseclone-odo --clone base
    $ source activate /ccsopen/home/${USER}/.conda/envs/baseclone-odo
    ```

* Deleting an environment:

    If for some reason you need to delete an environment, you can execute the following:

    ```bash
    $ conda env remove -p /path/to/your/env
    ```

* Exporting (sharing) an environment:

    You may want to share your environment with someone else.
    As mentioned previously, one way to do this is by creating your environment in a shared location where other users can access it.
    A different way (the method described below) is to export a list of all the packages and versions of your environment (an `environment.yml` file).
    If a different user provides conda the list you made, conda will install all the same package versions and recreate your environment for them -- essentially "sharing" your environment.
    To export your environment list:
    
    ```bash
    $ source activate my_env
    $ conda env export > environment.yml
    ```
    
    You can then email or otherwise provide the `environment.yml` file to the desired person.
    The person would then be able to create the environment like so:
    
    ```bash
    $ conda env create -f environment.yml
    ```

* Adding known environment locations:

    For a conda environment to be callable by a "name", it must be installed in one of the `envs_dirs` directories.
    The list of known directories can be seen by executing:

    ```bash
    $ conda config --show envs_dirs
    ```

    On Odo, the default location is your `$HOME` directory.
    If you plan to frequently create environments in a different location than the default (such as `/ccsopen/proj/...`), then there is an option to add directories to the `envs_dirs` list.
    To do so, you must execute:

    ```bash
    $ conda config --append envs_dirs /ccsopen/proj/<YOUR_PROJECT_ID>/${USER}/conda_envs/odo
    ```
    
    This will create a `.condarc` file in your `$HOME` directory if you do not have one already, which will now contain this new envs_dirs location.
    This will now enable you to use the `--name env_name` flag when using conda commands for environments stored in that specific directory, instead of having to use the `-p /ccsopen/proj/<YOUR_PROJECT_ID>/${USER}/conda_envs/odo/env_name` flag and specifying the full path to the environment.
    For example, you can do `source activate py39-odo` instead of `source activate /ccsopen/proj/<YOUR_PROJECT_ID>/${USER}/conda_envs/odo/py39-odo`.

&nbsp;

## Quick-Reference Commands

* List environments:

    ```bash
    $ conda env list
    ```

* List installed packages in current environment:

    ```bash
    $ conda list
    ```

* Creating an environment with Python version X.Y:

    For a **specific path**:

    ```bash
    $ conda create -p /path/to/your/my_env python=X.Y
    ```

    For a **specific name**:

    ```bash
    $ conda create -n my_env python=X.Y
    ```
       
* Deleting an environment:

    For a **specific path**:

    ```bash
    $ conda env remove -p /path/to/your/my_env
    ```

    For a **specific name**:

    ```bash
    $ conda env remove -n my_env
    ```

* Copying an environment:

    For a **specific path**:

    ```bash
    $ conda create -p /path/to/new_env --clone old_env
    ```

    For a **specific name**:

    ```bash
    $ conda create -n new_env --clone old_env
    ```
       
* Activating/Deactivating an environment:

    ```bash
    $ source activate my_env
    $ source deactivate # deactivates the current environment
    ```

* Installing/Uninstalling packages:

    Using **conda**:

    ```bash
    $ conda install package_name
    $ conda remove package_name
    ```

    Using **pip**:

    ```bash
    $ pip install package_name
    $ pip uninstall package_name
    $ pip install --no-binary=package_name package_name # builds from source
    ```

&nbsp;

## <a name="refs"></a>Additional Resources

* [Conda User Guide](https://conda.io/projects/conda/en/latest/user-guide/index.html)
* [Anaconda Package List](https://docs.anaconda.com/anaconda/packages/pkg-docs/)
* [Pip User Guide](https://pip.pypa.io/en/stable/user_guide/)
* [Using Pip In A Conda Environment](https://www.anaconda.com/blog/using-pip-in-a-conda-environment)
* [Python on OLCF Systems](https://docs.olcf.ornl.gov/software/python/index.html)
