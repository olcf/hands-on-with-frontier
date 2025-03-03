COMP  = amdclang
FLAGS = -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa  -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a
#COMP  = hipcc 
#FLAGS = -fopenmp --offload-arch=gfx90a
#FLAGS = -fopenmp -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a

INCLUDES  = -I/usr/include/openblas 
LIBRARIES = -lopenblas_serial

matrix_multiply: matrix_multiply.o
	${COMP} ${FLAGS} ${LIBRARIES} matrix_multiply.o -o matrix_multiply

matrix_multiply.o: matrix_multiply.c
	${COMP} ${FLAGS} ${INCLUDES} -c matrix_multiply.c

.PHONY: clean

clean:
	rm -f matrix_multiply *.o

