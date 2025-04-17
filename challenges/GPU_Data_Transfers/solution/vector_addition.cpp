#include "hip/hip_runtime.h"
#include <stdlib.h>
#include <stdio.h>

/* Size of array ========== */
#define N 1048576


/* Kernel ================= */
__global__ void add_vectors(double *a, double *b, double *c)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if(id < N) c[id] = a[id] + b[id];
}


/* Main program =========== */
int main()
{
    /* Number of bytes to allocate for N doubles ----------------------- */
    size_t bytes = N*sizeof(double);

    /* Allocate memory for arrays A, B, and C on host ------------------ */
    double *A = (double*)malloc(bytes);
    double *B = (double*)malloc(bytes);
    double *C = (double*)malloc(bytes);

    /* Allocate memory for arrays d_A, d_B, and d_C on device ---------- */
    double *d_A, *d_B, *d_C;
    hipMalloc(&d_A, bytes);
    hipMalloc(&d_B, bytes);
    hipMalloc(&d_C, bytes);

    /* Fill host arrays A and B ---------------------------------------- */
    for(int i=0; i<N; i++)
    {
        A[i] = 1.0;
        B[i] = 2.0;
    }

    /* Copy data from host arrays A and B to device arrays d_A and d_B - */
    /* TODO: Look up hipMemcpy API and..                                */
    /*  Replace the ?s in the hipMemcpy calls below with the correct    */
    /*  arguments to                                                     */
    /*     - copy host array A to device array d_A                       */
    /*     - copy host array B to device array d_B                       */
    /* ----------------------------------------------------------------- */
    hipMemcpy(d_A, A, bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_B, B, bytes, hipMemcpyHostToDevice);

    /* -------------------------------------------------------------------
    Set execution configuration parameters
        thr_per_wg: number of HIP threads per grid block
        wg_in_grid: number of workgroups in grid
    -------------------------------------------------------------------- */
    int thr_per_wg = 256;
    int wg_in_grid = ceil( float(N) / thr_per_wg );

    /* Launch kernel --------------------------------------------------- */
    /* hip add_vectors<<< wg_in_grid, thr_per_wg >>>(d_A, d_B, d_C); */
    /* 0,0 represents stream ID and shared memory size respectively. */
    /* Both are set to 0 meaning the default stream will be used     */
    /* with no shared memory allocation. This may change with        */
    /* asynchronous execution, but for now 0s are ok.                 */
     hipLaunchKernelGGL(add_vectors,wg_in_grid,thr_per_wg,0,0,d_A,d_B,d_C);

    /* Copy data from device array d_C to host array C ----------------- */
    /* TODO: Look up hipMemcpy API and...                               */
    /*  Replace the ?s in the hipMemcpy call below with the correct     */
    /*  arguments to                                                     */
    /*     - copy device array d_C to host array C                       */
    /* ----------------------------------------------------------------- */
    hipMemcpy(C, d_C, bytes, hipMemcpyDeviceToHost);

    /* Verify results -------------------------------------------------- */
    for(int i=0; i<N; i++)
    {
        if(C[i] != 3.0)
        { 
            printf("\nError: value of C[%d] = %d instead of 3.0\n\n", i, C[i]);
            exit(-1);
        }
    }	

    /* Free GPU memory ------------------------------------------------- */
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);

    /* Free CPU memory ------------------------------------------------- */
    free(A);
    free(B);
    free(C);

    /* Output ---------------------------------------------------------- */
    printf("\n---------------------------\n");
    printf("__SUCCESS__\n");
    printf("---------------------------\n");
    printf("N                 = %d\n", N);
    printf("Threads Per Block = %d\n", thr_per_wg);
    printf("Blocks In Grid    = %d\n", wg_in_grid);
    printf("---------------------------\n\n");

    return 0;
}
