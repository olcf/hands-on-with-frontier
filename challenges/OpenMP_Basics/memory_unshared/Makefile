CCOMP = gcc
CFLAGS = -fopenmp

vec_add_seq: vector_add_seq.o
	$(CCOMP) $(CFLAGS) vector_add_seq.o -o vec_add_seq

vector_add_seq.o: vector_add_seq.c
	$(CCOMP) $(CFLAGS) -c vector_add_seq.c

.PHONY: clean cleanall

clean:
	rm -f vec_add_seq *.o

cleanall:
	rm -f vec_add_seq *.o vec_add_test*