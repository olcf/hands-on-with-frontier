COMP = hipcc
CFLAGS = --offload-arch=gfx90a

INCLUDES  = -I${OLCF_OPENBLAS_ROOT}/include -I${ROCM_PATH}/include
LIBRARIES = -L${OLCF_OPENBLAS_ROOT}/lib -lopenblas  -L${ROCM_PATH}/lib -lhipblas -L${ROCM_PATH}/lib -lamdhip64

cpu_gpu_dgemm: cpu_gpu_dgemm.o
	${COMP} ${LFLAGS} ${LIBRARIES} cpu_gpu_dgemm.o -o cpu_gpu_dgemm

cpu_gpu_dgemm.o: cpu_gpu_dgemm.cpp
	${COMP} ${CFLAGS} ${INCLUDES} -c cpu_gpu_dgemm.cpp

.PHONY: clean

clean:
	rm -f cpu_gpu_dgemm *.o
