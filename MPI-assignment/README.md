###### Filip Stefaniuk 361039
# MPI Assignment
## Overview
This is implementation of simple molecular simulator that computes Axilrod-Teller potential. Program is written in `c`. There are available two versions of this program, sequential and distributed with MPI.

#### Project structure
Both implementations use common functions to handle i/o and update particle's position, velocity and acceleration. The only difference is algorithm for potential computation. Implementations of these algorithms (alongside main function) are in subfolders named `seq` and `dist/`.

#### Instalation
Program requires `gcc` copiler and `mpicc` compiler for distributed version.
After executing `make`, program named `body3` should be created. This is default distributed veersion. To compile sequential version use `make body3-seq`.

#### Execution
Start program with:
```
./body3 particles_in.txt particles_out stepcount deltatime [-v]

```
or 
```
mpiexec ./body3 particles_in.txt particles_out stepcount deltatime [-v] 
```
for distributed version, where:

- **particles_in.txt** defines the initial positions and velocities, the format is one particle per line, each line consists of 3 doubles specifying the x, y, z coordinates of a particle (single space separated); then 3 doubles specifying the vx, vy, vz velocities of a particle.

- **particles_out** is the base name of the output file. The actual result (same format as the input file) must be saved in a file particles_out_stepcount.txt.

- **stepcount** is the total number of steps of the Verlet algorithm.

- **deltatime** gives t between steps.

- **[-v]**  if present, puts the result after each step i
(counted from 1) in a file particles_out_i.txt

## Numerical Intensity