CC = mpicc
FLAGS= -O3 -lm
SRCS= ./src/particle.c ./src/utilities.c

.PHONY: body3-dist body3-seq clean

all: body3

body3:
	$(CC) $(FLAGS) -o $@ ./src/dist/main.c ./src/dist/simulation.c $(SRCS)

body3-seq:
	$(CC) $(FLAGS) -o $@ ./src/seq/main.c $(SRCS)

clean:
	rm -rf ./body3-seq ./body3