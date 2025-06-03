CCOMP = gcc
CFLAGS = -fopenmp

vec_add_par: vector_add_par.o
	$(CCOMP) $(CFLAGS) vector_add_par.o -o vec_add_par

vector_add_par.o: vector_add_par.c
	$(CCOMP) $(CFLAGS) -c vector_add_par.c

.PHONY: clean cleanall

clean:
	rm -f vec_add_par *.o

cleanall:
	rm -f vec_add_par *.o vec_add_test*