COMP = hipcc
CFLAGS = --offload-arch=gfx90a

INCLUDES  = -I/usr/include/openblas -I/opt/rocm/include/hipblas
LIBRARIES = -L/usr/lib/lib64 -lopenblas_serial  -L${ROCM_PATH}/lib -lhipblas

cpu_gpu_dgemm: cpu_gpu_dgemm.o
	${COMP} ${LFLAGS} ${LIBRARIES} cpu_gpu_dgemm.o -o cpu_gpu_dgemm

cpu_gpu_dgemm.o: cpu_gpu_dgemm.cpp
	${COMP} ${CFLAGS} ${INCLUDES} -c cpu_gpu_dgemm.cpp

.PHONY: clean

clean:
	rm -f cpu_gpu_dgemm *.o
