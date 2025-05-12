# Matrix Multiply on GPU Using hipBLAS

BLAS (Basic Linear Algebra Subprograms) are a set of linear algebra routines that perform basic vector and matrix operations on CPUs. The hipBLAS library includes a similar set of routines that perform basic linear algebra operations on GPUs. 

In this challenge, you will be given a program that initializes two matrices with random numbers, performs a matrix multiplication on the two matrices on the CPU, performs the same matrix multiplication on the GPU, then compares the results. The only part of the code that is missing is the call to `hipblasDgemm` that performs the GPU matrix multiplication. Your task will be to read the explanation for the `hipblasDgemm` routine below and then add a working `hipblasDgemm` call to the section of the code identified with a `TODO`.

The Dgemm function multiplies matrix A by matrix B to get product C. The Dgemm function also allows the user to multiply A and C by scalars α and β, which can be useful for maintaining stability when solving certain problems numerically. However, that is beyond the scope of this exercise, so you can assume that α = 1 and β = 0 for the purposes of this exercise.

To sum up, a Dgemm type function generally does this:
```
        C=α×A×B+β×C`
```
But since we have set α = 1 and β = 0, the problem you are solving is a basic matrix multiplication:
```       
        C=A×B
````

## The hipblasDgemm Call

The `hipblasDgemm` version of Dgemm in the [AMD hipblasDgemm documentation](https://hipblas.readthedocs.io/en/rocm-6.2.2/functions.html#list-of-level-3-blas-functions) is of the form:

```
hipblasDgemm(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, int k, const double *alpha, const double *AP, int lda, const double *BP, int ldb, const double *beta, double *CP, int ldc)
```

Let's break it down:

`hipblasHandle_t handle` : This variable contains the state of the function. If you wanted to, you could write a test based on this variable to see if the function completed its calculation successfully.

`hipblasOperation_t transA` and `hipblasOperation_t transB` : These arguments have specific options that control how you use matrices A and B. For example, you can use them as they are entered or use the transpose of either. Since we are solving C = A × B, we just want to use them as they are. The option for that is `HIPBLAS_OP_N`.

`int m`, `int n`, `int k` : These are the values of the dimensions of your matrices. 

`const double *alpha` : This is a pointer to the scalar alpha.

`const double *AP` : This is a pointer to matrix A on the GPU. 

`int lda` : This represents the value of the leading dimension of matrix A.

`const double *BP` : This is a pointer to matrix B on the GPU.

`int ldb` : This represents the value of the leading dimension of matrix B.

`const double *beta` : This is a pointer to the scalar beta.

`double *CP` : This is a pointer to the double-precision matrix C on the GPU.

`int ldc` : This represents the value of the leading dimension of matrix C.

Your job is to look at the code in `cpu_gpu_dgemm.cpp` and see if you can match the already existing variables to the arguments outlined above in the `hipblasDgemm` call. You must pay close attention to whether the varaibles are declared as pointers, doubles, or integers, and you must think about whether `hipblasDgemm` is expecting a varaible's value or its address in memory. A quick review of [Addresses and Pointers](https://github.com/olcf/foundational_hpc_skills/blob/master/intro_to_c/README.md#6-addresses-and-pointers) may be helpful before you start. 



## Add the Call to hipblasDgemm

Before getting started, you'll need to make sure you're in the `GPU_Matrix_Multiply/` directory:

```
$ cd ~/hands-on-with-frontier/challenges/GPU_Matrix_Multiply/
```

Look in the code `cpu_gpu_dgemm.cpp` and find the `TODO` section and add in the `hipblasDgemm` call.

> NOTE: You do not need to perform a transpose operation on the matrices, so the `hipblasOperation_t` arguments should be set to `HIPBLAS_OP_N`.

&nbsp;

## Compile the Code

Once you think you've correctly added the hipBLAS routine, try to compile the code.

First, you'll need to make sure your programming environment is set up correctly for this program. You'll need to use the cBLAS library for the CPU matrix multiply (`dgemm`) and the hipBLAS library for the GPU-version (`hipblasDgemm`), so you'll need to load the following modules:

```bash
$ module load PrgEnv-amd
$ module load rocm/$OLCF_FAMILY_COMPILER_VERSION             
$ module load openblas
```

Note: Our module system sets environment variables to help keep all the versions of the software you load consistent. In the example above, the `$OLCF_FAMILY_COMPILER_VERSION` environment variable holds the version of the AMD compiler that is loaded by module load `PrgEnv_amd`. We use this variable to ensure the same version of the ROCm GPU programming toolchain is loaded. 


Then, try to compile the code:

```bash
$ make
``` 

Did you encounter an error? If so, The compilation errors may assist in identifying the problem. 

## Run the Program

Once you've successfully compiled the code, try running it.

```bash
$ sbatch submit.sbatch
```

If the CPU and GPU give the same results, you will see the message `__SUCCESS__` in the output file. If you do not receive this message, try to identify the problem. As always, if you need help, make sure to ask.


### Hints

The hints get progressively more helpful as you go down. If you want to challenge yourself, you should only read as far as you need before attempting your next fix and compilation of the challenge code.  


* A good place to start is to observe how the variables declared in the code, map to the `cblasDgemm` arguments in the CPU version of Dgemm that is already correctly implemented in the code.
* If you are still unsure how the declared variables map to arguments in the functions, you may want to look up `cblasDgemm` and see how its arguments appear in the documentation, then compare those to the implemented `cblasDgemm` in the code.
* Next, look for the variable declarations made specifically for the GPU (device). Consider where those might fit in the `hipblasDgemm` arguments.
Remember that you do not need to perform a transpose operation on the matrices, so the `hipblasOperation_t` arguments should be set to `HIPBLAS_OP_N`.
* Pointers 

In the code we:
```
 /* Allocate memory for d_A, d_B, d_C on GPU ----------------------------------------*/
    double *d_A, *d_B, *d_C;'
```

Here: 
1.  `d_A`, `d_B`, `d_C` are declared as pointers on the GPU. In C, pointers are special variables used to store memory addresses. The `hipblasDgemm` function is looking for the *memory addresses*, not the *values*, for these pointers. 
See [Addresses and Pointers](https://github.com/olcf/foundational_hpc_skills/blob/master/intro_to_c/README.md#6-addresses-and-pointers) to determine if you should use the `d_A`, `*d_A`, or `&d_A` form of the variables to accomplish this.  


* Note that `hipblasDgemm` expects pointers for `alpha` and `beta`, but `alpha` and `beta` are declared as regular doubles for the CPU in the code. You must pass the addresses of `alpha` and `beta` in `hipblasDgemm`.
See [Addresses and Pointers](https://github.com/olcf/foundational_hpc_skills/blob/master/intro_to_c/README.md#6-addresses-and-pointers) to determine if you should use the `alpha`, `*alpha`, or `&alpha` form of the variables to accomplish this.


