CC = mpicc
FLAGS= -Wall -O3 -lm
SRCS= ./src/particle.c ./src/utilities.c

.PHONY: body3_dist body3_seq clean

all: body3

body3:
	$(CC) $(FLAGS) -o $@ ./src/dist/main.c ./src/dist/simulation.c $(SRCS)

body3-seq:
	$(CC) $(FLAGS) -o $@ ./src/seq/main.c $(SRCS)

clean:
	rm -rf ./body3-seq ./body3